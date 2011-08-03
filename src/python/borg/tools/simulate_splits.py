"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.simulate_splits import main

    plac.call(main)

import os.path
import csv
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

def simulate_split(maker, suite_path, train_paths, test_paths, suffix, budget):
    """Make a validation run."""

    suite = borg.fake.FakeSuite(borg.load_solvers(suite_path), test_paths, suffix)
    solver = maker(suite, train_paths, suffix)
    successes = 0

    for test_path in test_paths:
        logger.info("simulating run on %s", test_path)

        with suite.domain.task_from_path(test_path) as test_task:
            answer = solver(test_task)(budget)

            if suite.domain.is_final(test_task, answer):
                successes += 1

    return [len(train_paths), successes]

@plac.annotations(
    out_path = ("results output path"),
    portfolio_name = ("name of the portfolio to train"),
    suite_path = ("path to the solvers suite", "positional", None, os.path.abspath),
    budget = ("CPU seconds per instance", "positional", None, float),
    instances_root = ("path to instances", "positional", None, os.path.abspath),
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
    instances_root,
    model_name = "kernel",
    interval = 100.0,
    suffix = ".splits.csv",
    workers = 0,
    ):
    """Simulate portfolio and solver performance."""

    cargo.enable_default_logging()

    cargo.get_logger("borg.portfolios", level = "DETAIL")

    # generate jobs
    def yield_jobs():
        suite = borg.load_solvers(suite_path)
        paths = list(cargo.files_under(instances_root, suite.domain.extensions))
        maker = PortfolioMaker(portfolio_name, model_name, interval)

        logger.info("found %i instance(s) under %s", len(paths), instances_root)

        for _ in xrange(4):
            shuffled_paths = sorted(paths, key = lambda _ : random.random())

            #for left_size in xrange(10, 130, 20):
            for left_size in [1]:
                train_paths = shuffled_paths[:left_size]
                test_paths = shuffled_paths[-145:]

                yield (
                    simulate_split,
                    [maker, suite_path, train_paths, test_paths, suffix, budget],
                    )

    # and run them
    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["training", "solved"])

        cargo.do_or_distribute(yield_jobs(), workers, lambda _, r: writer.writerow(r))

