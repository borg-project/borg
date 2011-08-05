"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.evaluate_model import main

    plac.call(main)

import os.path
import csv
import random
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def simulate_split(model_name, suite_path, training, test_paths, suffix):
    """Make a validation run."""

    # build the portfolio
    logger.info("building portfolio given %i training runs", training.get_run_count())

    suite = borg.fake.FakeSuite(borg.load_solvers(suite_path), test_paths, suffix)

    if model_name == "multinomial":
        model = \
            borg.models.MultinomialModel.fit(
                suite.solvers,
                training,
                6000.0,
                len(training.run_lists),
                60,
                )
    elif model_name == "kernel":
        model = \
            borg.models.KernelModel.fit(
                suite.solvers,
                training,
                borg.models.DeltaKernel(),
                )
    else:
        raise Exception("unrecognized model name")

    portfolio = borg.portfolios.MixturePortfolio(suite, model)
    solver = borg.solver_io.RunningPortfolioFactory(portfolio, suite)

    # run the experiment
    budget = training.get_common_budget()
    successes = 0

    for test_path in test_paths:
        logger.info("simulating run on %s", test_path)

        with suite.domain.task_from_path(test_path) as test_task:
            answer = solver(test_task)(budget)

            if suite.domain.is_final(test_task, answer):
                successes += 1

    return [training.get_run_count(), successes]

def get_training_systematic(training, run_count):
    """Get a systematic subset of training data."""

    run_lists = sorted(training.run_lists.items(), key = lambda _: random.random())
    subset = borg.TrainingData()
    added = 0

    for (id_, runs) in run_lists:
        for run in sorted(runs, key = lambda _: random.random()):
            subset.add_run(id_, run)

            added += 1

            if added == run_count:
                return subset

@plac.annotations(
    out_path = ("results output path"),
    model_name = ("name of the model to train"),
    suite_path = ("path to the solvers suite", "positional", None, os.path.abspath),
    instances_root = ("path to instances", "positional", None, os.path.abspath),
    suffix = ("runs data file suffix", "option"),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(
    out_path,
    model_name,
    suite_path,
    instances_root,
    suffix = ".runs.csv",
    workers = 0,
    ):
    """Evaluate portfolio performance under a specified model."""

    cargo.enable_default_logging()

    def yield_jobs():
        suite = borg.load_solvers(suite_path)
        paths = list(cargo.files_under(instances_root, suite.domain.extensions))

        logger.info("found %i instance(s) under %s", len(paths), instances_root)

        for _ in xrange(4):
            shuffled_paths = sorted(paths, key = lambda _: random.random())
            split_size = int(len(paths) / 2)
            train_paths = shuffled_paths[:split_size]
            test_paths = shuffled_paths[split_size:]
            training = borg.TrainingData.from_paths(train_paths, suite.domain, suffix)

            for run_count in xrange(1, training.get_run_count(), 20):
                subset = get_training_systematic(training, run_count)

                yield (
                    simulate_split,
                    [model_name, suite_path, subset, test_paths, suffix],
                    )

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["training_runs", "solved"])

        cargo.do_or_distribute(yield_jobs(), workers, lambda _, r: writer.writerow(r))

