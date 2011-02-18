"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.run_validation2 import main

    plac.call(main)

import cargo

log = cargo.get_logger(__name__, default_level = "INFO")

def solve_cryptominisat(cnf_path, budget):
    """
    Run the CryptoMiniSat solver; report its cost and success.
    """

    seed = numpy.random.randint(0, 2**31)

    (stdout, stderr, code) = \
        cargo.call_capturing(
            [
                "/scratch/cluster/bsilvert/sat-competition-2011/solvers/run-1.4/run",
                "-k",
                "--time-limit={0}".format(int(round(budget))),
                "/scratch/cluster/bsilvert/sat-competition-2011/solvers/cryptominisat-2.9.0Linux64",
                "--randomize={0}".format(seed),
                cnf_path,
                ],
            )

    match = re.search(r"^c CPU time *: (\d+) *\(.+\)$", stdout, re.M)

    (cost,) = map(float, match.groups())

    #cpu_match = re.search(r"^\[run\] time:[ \t]*(\d+.\d+) seconds$", stderr, re.M)
    #(cpu_cost,) = map(float, cpu_match.groups())

    #logger.info("run took %.2f CPU seconds with seed %i", cpu_cost, seed)

    return (cost, code == 10)

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

@plac.annotations(
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    )
def main(tasks_root):
    """
    Run the script.
    """


# XXX solver, seed, budget, cost, success, answer

