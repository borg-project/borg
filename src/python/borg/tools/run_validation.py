"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.run_validation import main

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
            "--runs",
            type    = int,
            default = 1,
            metavar = "INT",
            help    = "make INT validation runs [%default]",
            ),
        Flag(
            "-f",
            "--training-fraction",
            type    = float,
            default = 0.0,
            metavar = "FLOAT",
            help    = "use fraction FLOAT of tasks for training [%default]",
            ),
        Flag(
            "--cache-path",
            metavar = "PATH",
            help    = "read data from PATH when possible [%default]",
            ),
        )

def make_validation_run(
    engine_url,
    request,
    domain,
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
    from borg.portfolio.world import build_trainer
    from borg.solvers         import solver_from_request

    trainer = build_trainer(domain, train_uuids, CacheSession, extrapolation = 6)
    solver  = solver_from_request(trainer, request)

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
            components = request["strategy"]["model"]["components"]
            model_type = request["strategy"]["model"]["type"]
        else:
            components = None
            model_type = None

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
                )

        session.add(run)
        session.commit()

def yield_solver_requests():
    """
    Build the solvers as configured.
    """

    import numpy

    sat_2009_subsolvers = [
        "sat/2009/adaptg2wsat2009++",
        "sat/2009/CirCUs",
        "sat/2009/clasp",
        "sat/2009/glucose",
        "sat/2009/gnovelty+2",
        "sat/2009/gNovelty+-T",
        "sat/2009/hybridGM3",
        "sat/2009/iPAWS",
        "sat/2009/IUT_BMB_SAT",
        "sat/2009/LySAT_c",
        "sat/2009/LySAT_i",
        "sat/2009/ManySAT",
        "sat/2009/march_hi",
        "sat/2009/minisat_09z",
        "sat/2009/minisat_cumr_p",
        "sat/2009/mxc_09",
        "sat/2009/precosat",
        "sat/2009/rsat_09",
        "sat/2009/SApperloT",
        "sat/2009/TNM",
        "sat/2009/VARSAT-industrial"
        ]
    sat_2009_satzillas = [
        "sat/2009/SATzilla2009_R",
        "sat/2009/SATzilla2009_C",
        "sat/2009/SATzilla2009_I",
        ]

    # the individual solvers
#     for name in sat_2009_subsolvers + sat_2009_satzillas:
#         yield { "type" : "lookup", "name" : name }

    # the DCM portfolio solver(s)
#     for k in xrange(1, 65):
    for k in [63]:
        yield {
            "type"     : "portfolio",
            "strategy" : {
                "type"     : "modeling",
                "model"    : {
                    "type"        : "dcm",
                    "components"  : int(k),
                    "em_restarts" : 4,
                    "actions"     : {
                        "solvers" : sat_2009_subsolvers,
                        "budgets" : numpy.r_[25.0:4000.0:10j].tolist(),
                        },
                    },
                "planner" : {
                    "type"     : "hard_myopic",
                    "discount" : 1.0 - 1e-4,
                    },
                },
            }

    # the multinomial portfolio solver(s)
#     for k in xrange(1, 65):
#         yield {
#             "type"     : "portfolio",
#             "strategy" : {
#                 "type"     : "modeling",
#                 "model"    : {
#                     "type"        : "multinomial",
#                     "components"  : int(k),
#                     "em_restarts" : 4,
#                     "actions"     : {
#                         "solvers" : sat_2009_subsolvers,
#                         "budgets" : numpy.r_[25.0:4000.0:10j].tolist(),
#                         },
#                     },
#                 "planner" : {
#                     "type"     : "hard_myopic",
#                     "discount" : 1.0 - 1e-4,
#                     },
#                 },
#             }

def main():
    """
    Run the script.
    """

    # get command line arguments
    import cargo.labor.storage
    import borg.data
    import borg.tasks
    import borg.solvers

    from uuid           import UUID
    from os.path        import abspath
    from cargo.json     import load_json
    from cargo.flags    import parse_given
    from cargo.temporal import TimeDelta

    (group, budget, fraction, uuids) = \
        parse_given(usage = "%prog [options] <group> <budget> <fraction> <uuids.json>")

    budget   = TimeDelta(seconds = float(budget))
    fraction = float(fraction)
    uuids    = map(UUID, load_json(uuids))

    if module_flags.given.cache_path is None:
        cache_path = None
    else:
        cache_path = abspath(module_flags.given.cache_path)

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
                    for i in xrange(module_flags.given.runs):
                        yield CallableJob(
                            make_validation_run,
                            engine_url    = session.connection().engine.url,
                            request       = request,
                            domain        = "sat",
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

