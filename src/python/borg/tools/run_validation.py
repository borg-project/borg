"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                      import call
    from borg.tools.run_validation import main

    call(main)

from uuid       import UUID
from datetime   import parse_timedelta
from plac       import annotations
from cargo.log  import get_logger
from cargo.json import load_json

log = get_logger(__name__, default_level = "INFO")

def make_validation_run(
    engine_url,
    request,
    fraction,
    task_uuids,
    budget,
    random,
    named_solvers,
    group,
    cache_path,
    ):
    """
    Train and test a solver.
    """

    # make sure that we're logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("cargo.statistics.mixture", level = "DETAIL")

    # generate train-test splits
    from cargo.iterators import shuffled

    shuffled_uuids = shuffled(task_uuids, random)
    len_train      = int(round(fraction * len(shuffled_uuids)))
    train_uuids    = shuffled_uuids[:len_train]
    test_uuids     = shuffled_uuids[len_train:]

    log.info("split tasks into %i training and %i test", len(train_uuids), len(test_uuids))

    # connect to the database
    from cargo.sql.alchemy import (
        SQL_Engines,
        make_session,
        )

    main_engine = SQL_Engines.default.get(engine_url)
    MainSession = make_session(bind = main_engine)

    # retrieve the local cache
    from cargo.io import cache_file

    if cache_path is None:
        CacheSession = MainSession
    else:
        cache_engine = SQL_Engines.default.get("sqlite:///%s" % cache_file(cache_path))
        CacheSession = make_session(bind = cache_engine)

    # build the solver
    from borg.solvers         import AbstractSolver
    from borg.portfolio.world import Trainer

    trainer   = DecisionTrainer.build(ResearchSession, train_uuids, request["trainer"])
    requested = AbstractSolver.build(trainer, request["solver"])

    log.info("built solver from request")

    with MainSession() as session:
        # set up the environment
        from borg.solvers import Environment

        environment = \
            Environment(
                MainSession   = MainSession,
                CacheSession  = CacheSession,
                named_solvers = named_solvers,
                )

        # run over specified tasks
        solved = 0

        for test_uuid in test_uuids:
            from borg.data  import TaskRow
            from borg.tasks import Task

            task_row = session.query(TaskRow).get(test_uuid)
            task     = Task(row = task_row)

            session.commit()

            # run on this task
            attempt = solver.solve(task, budget, random, environment)

            if attempt.answer is not None:
                solved += 1

            log.info("ran on %s (success? %s)", task_row.uuid, attempt.answer is not None)

        # store the result
        from borg.data import ValidationRunRow

        log.info("solver succeeded on %i of %i task(s)", solved, len(test_uuids))

        if solver.name == "portfolio":
            components    = request["solver"]["strategy"]["model"]["components"]
            model_type    = request["solver"]["strategy"]["model"]["type"]
            analyzer_type = request["solver"]["analyzer"]["type"]
        else:
            components    = None
            model_type    = None
            analyzer_type = None

        run = \
            ValidationRunRow(
                solver           = solver.get_row(session),
                solver_request   = request,
#                 train_task_uuids = train_uuids,
#                 test_task_uuids  = test_uuids,
                group            = group,
                score            = solved,
                components       = components,
                model_type       = model_type,
                analyzer_type    = analyzer_type,
                )

        session.add(run)
        session.commit()

def outsource_validation_jobs(
    engine_url,
    builders,
    task_uuids,
    budget,
    fraction,
    group = None,
    repeats = 1,
    ):
    """
    Outsource validation runs.
    """

    # build its jobs
    def yield_jobs():
        from cargo.labor.jobs import CallableJob
        from cargo.random     import get_random_random
        from borg.solvers     import get_named_solvers

        for builder in builders:
            for i in xrange(repeats):
                yield CallableJob(
                    make_validation_run,
                    engine_url    = engine_url,
                    builder       = builder,
                    fraction      = fraction,
                    task_uuids    = task_uuids,
                    budget        = budget,
                    random        = get_random_random(),
                    named_solvers = get_named_solvers(use_recycled = True),
                    group         = group,
                    cache_path    = None,
                    )

    jobs = list(yield_jobs())

    # run the jobs
    from cargo.temporal      import utc_now
    from cargo.labor.storage import outsource_or_run

    outsource_or_run(jobs, "validation runs (at %s)" % utc_now())

@annotations(
    group        = ("group name")       ,
    budget       = ("solver time budget", "positional", None , parse_timedelta),
    fraction     = ("train set fraction", "positional", None , float)          ,
    uuids        = ("task uuids json"   , "positional", None , lambda p: map(UUID, load_json(p))),
    trial        = ("attempts trial"    , "option"    , "t"  , UUID)           ,
    parent_trial = ("trial parent"      , "option"    , "p"  , UUID)
    runs         = ("number of runs"    , "option"    , "r"  , int)            ,
    cache_path   = ("local database"    , "option"    , None),
    )
def main(
    group,
    budget,
    fraction,
    uuids,
    trial        = "random",
    parent_trial = None,
    runs         = 1,
    cache_path   = None,
    ):
    """
    Run the script.
    """

    if cache_path is not None:
        from os.path import abspath

        cache_path = abspath(cache_path)

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine",        level = "WARNING")
    get_logger("cargo.unix.accounting",    level = "WARNING")
    get_logger("borg.solvers.competition", level = "NOTE")

    # connect to the database and go
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        from cargo.sql.alchemy import make_session
        from borg.data         import research_connect

        ResearchSession = make_session(bind = research_connect())

        with ResearchSession() as session:
            # build its jobs
            def yield_jobs():
                """
                Generate a set of jobs to distribute.
                """

                from cargo.labor.jobs import CallableJob
                from cargo.random     import get_random_random
                from borg.solvers     import get_named_solvers

                named_solvers = get_named_solvers(use_recycled = True)

                for request in yield_solver_requests():
                    for i in xrange(runs):
                        yield CallableJob(
                            make_validation_run,
                            engine_url    = session.connection().engine.url,
                            request       = request,
                            fraction      = fraction,
                            task_uuids    = uuids,
                            budget        = budget,
                            random        = get_random_random(),
                            named_solvers = named_solvers,
                            group         = group,
                            cache_path    = cache_path,
                            )

            jobs = list(yield_jobs())

        # run the jobs
        from cargo.temporal      import utc_now
        from cargo.labor.storage import outsource_or_run

        outsource_or_run(jobs, "validation runs (at %s)" % utc_now())

# XXX solver, seed, budget, cost, success, answer

