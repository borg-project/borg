"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                   import call
    from borg.tools.run_solvers import main

    call(main)

from plac           import annotations
from cargo.log      import get_logger
from cargo.temporal import parse_timedelta

log = get_logger(__name__, default_level = "INFO")

def solve_task(
    engine_url,
    trial_row,
    solver,
    task_uuid,
    budget,
    random,
    named_solvers,
    collections,
    use_recycled,
    ):
    """
    Make some number of solver runs on a task.
    """

    # make sure that we're logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("cargo.unix.accounting",      level = "DEBUG")
    get_logger("borg.solvers.competition",   level = "DEBUG")
    get_logger("borg.solvers.uncompressing", level = "DEBUG")

    # connect to the database
    from cargo.sql.alchemy import (
        SQL_Engines,
        make_session,
        )

    main_engine = SQL_Engines.default.get(engine_url)
    MainSession = make_session(bind = main_engine)

    with MainSession() as session:
        # set up the environment
        from borg.solvers import Environment

        environment = \
            Environment(
                MainSession   = MainSession,
                CacheSession  = MainSession, # FIXME
                named_solvers = named_solvers,
                collections   = collections,
                )

        # prepare the run
        import borg.solvers.base

        from borg.data import TaskRow

        trial_row = session.merge(trial_row)
        task_row  = session.query(TaskRow).get(task_uuid)

        if use_recycled:
            from borg.tasks import Task

            full_solver = solver
            task        = Task(row = task_row)
        else:
            from borg.solvers import UncompressingSolver

            full_solver = UncompressingSolver(solver)
            task        = task_row.get_task(environment)

        # make the run
        log.info("running %s on %s", solver.name, task_row.uuid)

        session.commit()

        attempt = full_solver.solve(task, budget, random, environment)

        # store the attempt
        run_row = attempt.get_row(session)

        run_row.trials = [trial_row]

        session.commit()

def yield_solvers(session, solver_pairs):
    """
    Build the solvers as configured.
    """

    from borg.solvers import LookupSolver

    if solver_pairs is None:
        from borg.data import SolverRow as SR

        for (name,) in session.query(SR.name):
            yield LookupSolver(name)
    else:
        for (kind, name) in solver_pairs:
            if kind == "name":
                yield LookupSolver(name)
            elif kind == "load":
                import cPickle as pickle

                from cargo.io import expandpath

                with open(expandpath(name)) as file:
                    yield pickle.load(file)
            else:
                raise ValueError("unknown solver kind")

def yield_task_uuids(session, task_uuids):
    """
    Look up or return the task uuids.
    """

    if task_uuids is None:
        from borg.data import TaskRow as TR

        for (uuid,) in session.query(TR.uuid):
            yield uuid
    else:
        from uuid import UUID

        for s in task_uuids:
            yield UUID(s)

@annotations(
    budget          = ("solver budget"    , "positional", None , parse_timedelta),
    arguments       = ("arguments in JSON", "positional", None),
    trial           = ("place in trial"   , "option"    , "t"  , UUID)           ,
    parent_trial    = ("with parent trial", "option"    , "p"  , UUID)           ,
    restarts        = ("minimum attempts" , "option"    , "r"  , int)            ,
    seeded_restarts = ("minimum attempts" , "option"    , "s"  , int)            ,
    recycle         = ("reuse past runs"  , "flag")     ,
    )
def main(
    budget,
    arguments       = None,
    trial           = "random",
    parent_trial    = None,
    restarts        = 1,
    seeded_restarts = 1,
    recycle         = False,
    ):
    """
    Run the script.
    """

    # get arguments
    from cargo.json import load_json

    if arguments is None:
        arguments = {}
    else:
        arguments = load_json(arguments)

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "WARNING")

    # connect to the database and go
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        from cargo.sql.alchemy import make_session
        from borg.data         import research_connect

        ResearchSession = make_session(bind = research_connect())

        with ResearchSession() as session:
            # create a trial
            from cargo.temporal import utc_now
            from borg.data      import TrialRow

            trial_label = "solver runs (at %s)" % utc_now()

            if parent_trial is None:
                parent_trial = None
            else:
                parent_trial = session.query(TrialRow).get(parent_trial)

                assert parent_trial is not None

            if trial == "random":
                trial_row = TrialRow(label = trial_label, parent = parent_trial)

                session.add(trial_row)
            else:
                assert parent_trial is None

                trial_row = session.query(TrialRow).get(trial)

                assert trial_row is not None

            session.commit()

            log.note("placing attempts in trial %s", trial_row.uuid)

            # build its jobs
            def yield_jobs():
                """
                Generate a set of jobs to distribute.
                """

                from cargo.labor.jobs import CallableJob
                from cargo.random     import get_random_random
                from borg.tasks       import get_collections
                from borg.solvers     import (
                    Environment,
                    get_named_solvers,
                    )

                named_solvers = get_named_solvers(use_recycled = recycle)
                environment   = Environment(named_solvers = named_solvers)
                collections   = get_collections()

                for solver in yield_solvers(session, arguments.get("solvers")):
                    if solver.get_seeded(environment):
                        restarts = max(restarts, seeded_restarts)

                    log.info("making %i restarts of %s", restarts, solver.name)

                    for task_uuid in yield_task_uuids(session, arguments.get("tasks")):
                        for i in xrange(restarts):
                            yield CallableJob(
                                solve_task,
                                engine_url    = session.connection().engine.url,
                                trial_row     = trial_row,
                                solver        = solver,
                                task_uuid     = task_uuid,
                                budget        = budget,
                                random        = get_random_random(),
                                named_solvers = named_solvers,
                                collections   = collections,
                                use_recycled  = recycle,
                                )

            jobs = list(yield_jobs())

        # run the jobs
        from cargo.labor.storage import outsource_or_run

        outsource_or_run(jobs, trial_label)

