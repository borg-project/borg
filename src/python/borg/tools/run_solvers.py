"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.run_solvers import main

    raise SystemExit(main())

from cargo.log   import get_logger
from cargo.flags import (
    Flag,
    Flags,
    )

log          = get_logger(__name__, default_level = "INFO")
module_flags = \
    Flags(
        "Script Options",
        Flag(
            "-t",
            "--trial",
            default = "random",
            metavar = "UUID",
            help    = "place attempts in trial UUID [%default]",
            ),
        Flag(
            "-p",
            "--parent-trial",
            default = None,
            metavar = "UUID",
            help    = "use a child trial of UUID [%default]",
            ),
        Flag(
            "-r",
            "--restarts",
            type    = int,
            default = 1,
            metavar = "INT",
            help    = "make INT restarts [%default]",
            ),
        )

def solve_task(
    engine_url,
    trial_row,
    solver,
    task_uuid,
    budget,
    random,
    named_solvers,
    collections,
    ):
    """
    Make some number of preprocessor runs on a task.
    """

    # make sure that we're logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("borg.solvers.satelite", level = "DEBUG")

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
                named_solvers = named_solvers,
                collections   = collections,
                )

        # prepare the run
        from borg.data    import TaskRow
        from borg.solvers import UncompressingSolver

        full_solver = UncompressingSolver(solver)
        trial_row   = session.merge(trial_row)
        task_row    = session.query(TaskRow).get(task_uuid)
        task        = task_row.get_task(environment)

        # make the run
        log.info("running %s on %s", solver.name, task_row.uuid)

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

    for (kind, name) in solver_pairs:
        if kind == "name":
            yield LookupSolver(name)
        elif kind == "solver":
            with open(name) as file:
                yield pickle.load(file)
        else:
            raise ValueError("unknown solver kind")

def yield_jobs(session, trial_row, budget, solver_pairs, task_uuids, restarts, collections):
    """
    Generate a set of jobs to distribute.
    """

    # yield jobs
    from cargo.labor.jobs import CallableJob
    from cargo.random     import get_random_random
    from borg.solvers     import get_named_solvers

    named_solvers = get_named_solvers()

    for solver in yield_solvers(session, solver_pairs):
        for task_uuid in task_uuids:
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
                    )

def main():
    """
    Run the script.
    """

    # get command line arguments
    import cargo.labor.storage
    import borg.data
    import borg.tasks
    import borg.solvers

    from cargo.json     import load_json
    from cargo.flags    import parse_given
    from cargo.temporal import TimeDelta

    (budget, arguments) = parse_given(usage = "%prog <budget> <args.json> [options]")

    budget    = TimeDelta(seconds = float(budget))
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

            if module_flags.given.parent_trial is None:
                parent_trial = None
            else:
                parent_trial = session.query(TrialRow).get(module_flags.given.parent_trial)

                assert parent_trial is not None

            if module_flags.given.trial == "random":
                trial_row = TrialRow(label = trial_label, parent = parent_trial)

                session.add(trial_row)
            else:
                assert parent_trial is None

                trial_row = session.query(TrialRow).get(module_flags.given.trial)

                assert trial_row is not None

            session.commit()

            # build jobs
            from uuid       import UUID
            from borg.tasks import get_collections

            jobs = \
                list(
                    yield_jobs(
                        session,
                        trial_row,
                        budget,
                        arguments["solvers"],
                        map(UUID, arguments["tasks"]),
                        module_flags.given.restarts,
                        get_collections(),
                        ),
                    )

        # run the jobs
        from cargo.labor.storage import outsource_or_run

        outsource_or_run(jobs, trial_label)

