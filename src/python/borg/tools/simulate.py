"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.simulate import main

    plac.call(main)

import os.path
import csv
import uuid
import random
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

class PortfolioMaker(object):
    def __init__(self, portfolio_name, model_name, interval):
        self.name = portfolio_name
        self.model_name = model_name
        self.interval = interval

    def __call__(self, suite, train_paths, suffix):
        training = borg.storage.TrainingData(train_paths, suite.domain, suffix = suffix)
        factory = borg.portfolios.named[self.name]
        portfolio = factory(suite, training, self.interval)

        return borg.solver_io.RunningPortfolioFactory(portfolio, suite)

class SolverMaker(object):
    def __init__(self, solver_name):
        self.name = solver_name

    def __call__(self, suite, train_paths, suffix):
        return suite.solvers[self.name]

def simulate_split(maker, suite_path, train_paths, test_paths, suffix, budget, split):
    """Make a validation run."""

    suite = borg.fake.FakeSuite(borg.load_solvers(suite_path), test_paths, suffix)
    solver = maker(suite, train_paths, suffix)
    rows = []

    for test_path in test_paths:
        logger.info("simulating run on %s", test_path)

        with suite.domain.task_from_path(test_path) as test_task:
            with borg.accounting() as accountant:
                answer = solver(test_task)(budget)

            succeeded = suite.domain.is_final(test_task, answer)
            cpu_cost = accountant.total.cpu_seconds

            logger.info(
                "%s %s on %s (%.2f CPU s)",
                maker.name,
                "succeeded" if succeeded else "failed",
                os.path.basename(test_path),
                cpu_cost,
                )

            success_str = "TRUE" if succeeded else "FALSE"

            rows.append([test_path, maker.name, budget, cpu_cost, success_str, None, split])

            #if not succeeded:
                #feasible = False

                #for run in suite.runs_data.get_run_list(test_path):
                    #logger.info("%s: %s in %.2f", run.solver, run.success, run.cost)

                    #feasible = feasible or run.success

                #if feasible:
                    #raise SystemExit()

    return rows

def yield_split_runs(makers, suite, budget):
    # assemble task set
    paths = list(cargo.files_under(tasks_root, suite.domain.extensions))
    examples = int(round(len(paths) * 0.50))

    logger.info("found %i tasks under %s", len(paths), tasks_root)

    # build validation runs
    for _ in xrange(runs):
        split = uuid.uuid4()
        shuffled = sorted(paths, key = lambda _ : random.random())
        train_paths = shuffled[:examples]
        test_paths = shuffled[examples:]

        for maker in makers:
            yield (
                simulate_split,
                [maker, suite_path, train_paths, test_paths, suffix, budget, split],
                )

def yield_explicit_runs(makers, splits, suite_path, suffix, budget):
    """Make runs on train/test splits."""

    for (train_paths, test_paths) in splits:
        split_id = uuid.uuid4()

        for maker in makers:
            yield (
                simulate_split,
                [maker, suite_path, train_paths, test_paths, suffix, budget, split_id],
                )

@plac.annotations(
    out_path = ("results output path"),
    portfolio_name = ("name of the portfolio to train"),
    suite_path = ("path to the solvers suite", "positional", None, os.path.abspath),
    budget = ("CPU seconds per instance", "positional", None, float),
    train_root = ("path to train task files", "positional", None, os.path.abspath),
    test_root = ("path to test task files", "positional", None, os.path.abspath),
    model_name = ("name of portfolio model", "option", None),
    interval = ("planner discretization width", "option", None),
    suffix = ("runs data file suffix", "option"),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(
    out_path,
    portfolio_name,
    suite_path,
    budget,
    train_root,
    test_root,
    model_name = "kernel",
    interval = 100.0,
    suffix = ".runs.csv",
    workers = 0,
    ):
    """Simulate portfolio and solver performance."""

    cargo.enable_default_logging()

    cargo.get_logger("borg.portfolios", level = "DETAIL")

    # prepare solver makers
    suite = borg.load_solvers(suite_path)

    if portfolio_name == "-":
        makers = map(SolverMaker, suite.solvers)
    else:
        makers = [PortfolioMaker(portfolio_name, model_name, interval)]

    # generate jobs
    train_paths = list(cargo.files_under(train_root, suite.domain.extensions))
    test_paths = list(cargo.files_under(test_root, suite.domain.extensions))

    logger.info("found %i train task(s) under %s", len(train_paths), train_root)
    logger.info("found %i test task(s) under %s", len(test_paths), test_root)

    splits = [(train_paths, test_paths)]
    jobs = list(yield_explicit_runs(makers, splits, suite_path, suffix, budget))

    # and run them
    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["path", "solver", "budget", "cost", "success", "answer", "split"])

        cargo.do_or_distribute(jobs, workers, lambda _, r: writer.writerows(r))

