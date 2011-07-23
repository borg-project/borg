"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.simulate import main

    plac.call(main)

import os.path
import csv
import uuid
import itertools
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

class PortfolioMaker(object):
    def __init__(self, portfolio_name):
        self.name = portfolio_name

    def __call__(self, suite, train_paths):
        training = borg.storage.TrainingData(train_paths, suite.domain, suffix = ".ppfolio.runs.csv")
        factory = borg.portfolios.named[self.name]
        portfolio = factory(suite, training, 100.0, 50) # XXX

        return borg.solver_io.RunningPortfolioFactory(portfolio, suite)

class SolverMaker(object):
    def __init__(self, solver_name):
        self.name = solver_name

    def __call__(self, suite, train_paths):
        return suite.solvers[self.name]

def simulate_split(maker, suite_path, train_paths, test_paths, suffix, budget, split):
    """Make a validation run."""

    suite = borg.fake.FakeSuite(borg.load_solvers(suite_path), test_paths, suffix)
    solver = maker(suite, train_paths)
    successes = []

    for test_path in test_paths:
        logger.info("simulating run on %s", test_path)

        with suite.domain.task_from_path(test_path) as test_task:
            with borg.accounting() as accountant:
                answer = solver(test_task)(budget)

            succeeded = suite.domain.is_final(test_task, answer)
            cpu_cost = accountant.total.cpu_seconds

            if succeeded:
                successes.append(cpu_cost)

            logger.info(
                "%s %s on %s (%.2f CPU s)",
                maker.name,
                "succeeded" if succeeded else "failed",
                os.path.basename(test_path),
                cpu_cost,
                )

    logger.info(
        "method %s had final success rate %.2f",
        maker.name,
        float(len(successes)) / len(test_paths),
        )

    return \
        zip(
            itertools.repeat(maker.name),
            itertools.repeat(budget),
            sorted(successes),
            numpy.arange(len(successes) + 1.0),
            itertools.repeat(split),
            )

@plac.annotations(
    out_path = ("results output path"),
    portfolio_name = ("name of the portfolio to train"),
    suite_path = ("path to the solvers suite"),
    budget = ("CPU seconds per instance", "positional", None, float),
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    suffix = ("runs data file suffix", "option"),
    runs = ("number of validation runs", "option", "r", int),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(
    out_path,
    portfolio_name,
    suite_path,
    budget,
    tasks_root,
    suffix = ".runs.csv",
    runs = 8,
    workers = 0,
    ):
    """Simulate portfolio and solver performance."""

    cargo.enable_default_logging()

    cargo.get_logger("borg.portfolios", level = "DETAIL")

    def yield_runs():
        # assemble task set
        suite = borg.load_solvers(suite_path)
        paths = list(cargo.files_under(tasks_root, suite.domain.extensions))
        examples = int(round(len(paths) * 0.50))

        logger.info("found %i tasks under %s", len(paths), tasks_root)

        # prepare solver makers
        if portfolio_name == "-":
            makers = map(SolverMaker, suite.solvers)
        else:
            makers = [PortfolioMaker(portfolio_name)]

        # build validation runs
        for _ in xrange(runs):
            split = uuid.uuid4()
            shuffled = sorted(paths, key = lambda _ : numpy.random.rand())
            train_paths = shuffled[:examples]
            test_paths = shuffled[examples:]

            for maker in makers:
                yield (
                    simulate_split,
                    [maker, suite_path, train_paths, test_paths, suffix, budget, split],
                    )

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["name", "budget", "cost", "solved", "split"])

        cargo.do_or_distribute(yield_runs(), workers, lambda _, r: writer.writerows(r))

