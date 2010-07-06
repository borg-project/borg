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
            )
        )

def make_validation_run(
    engine_url,
    trial_row,
    request,
    domain,
    fraction,
    task_uuids,
    budget,
    random,
    named_solvers,
    ):
    """
    Train and test a solver.
    """

    # make sure that we're logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    # generate train-test splits
    from cargo.iterators import shuffled

    shuffled_uuids = shuffled(task_uuids, random)
    len_train      = int(fraction * len(shuffled_uuids))
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

    with MainSession() as session:
        # set up the environment
        from borg.solvers import Environment

        environment = \
            Environment(
                MainSession   = MainSession,
                CacheSession  = MainSession, # FIXME
                named_solvers = named_solvers,
                )

        # build the solver
        from borg.portfolio.world import build_trainer
        from borg.solvers         import solver_from_request

        trainer = build_trainer(domain, train_uuids, environment.CacheSession)
        solver  = solver_from_request(request, trainer)

        log.info("built solver from request")

        # run over specified tasks
        trial_row = session.merge(trial_row)
        solved    = 0

        for test_uuid in test_uuids:
            from borg.data  import TaskRow
            from borg.tasks import Task

            task_row = session.query(TaskRow).get(test_uuid)
            task     = Task(row = task_row)

            # run on this task
            attempt = solver.solve(task, budget, random, environment)

            if attempt.answer is not None:
                solved += 1

            log.info("ran on %s (success? %s)", task_row.uuid, attempt.answer is not None)

        # store the attempt
        log.info("solver succeeded on %i of %i task(s)", solved, len(test_uuids))

        # FIXME store result

def yield_solver_requests():
    """
    Build the solvers as configured.
    """

#     yield { "type" : "lookup", "name" : "sat/2009/CirCUs" }
#     yield { "type" : "lookup", "name" : "sat/2009/adaptg2wsat2009++" }
#     yield { "type" : "lookup", "name" : "sat/2009/CirCUs" }
#     yield { "type" : "lookup", "name" : "sat/2009/clasp" }
#     yield { "type" : "lookup", "name" : "sat/2009/glucose" }
#     yield { "type" : "lookup", "name" : "sat/2009/gnovelty+2" }
#     yield { "type" : "lookup", "name" : "sat/2009/gNovelty+-T" }
#     yield { "type" : "lookup", "name" : "sat/2009/hybridGM3" }
#     yield { "type" : "lookup", "name" : "sat/2009/iPAWS" }
#     yield { "type" : "lookup", "name" : "sat/2009/IUT_BMB_SAT" }
#     yield { "type" : "lookup", "name" : "sat/2009/LySAT_c" }
#     yield { "type" : "lookup", "name" : "sat/2009/LySAT_i" }
#     yield { "type" : "lookup", "name" : "sat/2009/ManySAT" }
#     yield { "type" : "lookup", "name" : "sat/2009/march_hi" }
#     yield { "type" : "lookup", "name" : "sat/2009/minisat_09z" }
#     yield { "type" : "lookup", "name" : "sat/2009/minisat_cumr_p" }
#     yield { "type" : "lookup", "name" : "sat/2009/mxc_09" }
#     yield { "type" : "lookup", "name" : "sat/2009/precosat" }
#     yield { "type" : "lookup", "name" : "sat/2009/rsat_09" }
    yield { "type" : "lookup", "name" : "sat/2009/SApperloT" }
#     yield { "type" : "lookup", "name" : "sat/2009/SATzilla2009_C" }
#     yield { "type" : "lookup", "name" : "sat/2009/SATzilla2009_I" }
#     yield { "type" : "lookup", "name" : "sat/2009/SATzilla2009_R" }
#     yield { "type" : "lookup", "name" : "sat/2009/TNM" }
#     yield { "type" : "lookup", "name" : "sat/2009/VARSAT-industrial" }

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
    from cargo.json     import load_json
    from cargo.flags    import parse_given
    from cargo.temporal import TimeDelta

    (budget, fraction, uuids) = parse_given(usage = "%prog [options] <budget> <fraction> <uuids.json>")

    budget   = TimeDelta(seconds = float(budget))
    fraction = float(fraction)
    uuids    = map(UUID, load_json(uuids))

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
            # create a trial
            from cargo.temporal import utc_now
            from borg.data      import TrialRow

            trial_row = \
                TrialRow.as_specified(
                    session,
                    module_flags.given.trial,
                    module_flags.given.parent_trial, 
                    "validation runs (at %s)" % utc_now(),
                    )

            session.flush()

            log.note("placing attempts in trial %s", trial_row.uuid)

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
                            trial_row     = trial_row,
                            request       = request,
                            domain        = "sat",
                            fraction      = fraction,
                            task_uuids    = uuids,
                            budget        = budget,
                            random        = get_random_random(),
                            named_solvers = named_solvers,
                            )

            jobs = list(yield_jobs())

        # run the jobs
        from cargo.labor.storage import outsource_or_run

        outsource_or_run(jobs, trial_row.label)

