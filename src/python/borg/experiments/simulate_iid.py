"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import numpy
import sklearn
import condor
import borg

from borg.experiments.simulate_runs import PortfolioMaker

logger = borg.get_logger(__name__, default_level = "INFO")

def simulate_run(run, maker, all_data, train_mask, test_mask, instances):
    """Simulate portfolio execution on a train/test split."""

    train_data = all_data.masked(train_mask)
    test_data = all_data.masked(test_mask)
    budget = test_data.common_budget
    suite = borg.fake.FakeSuite(test_data)
    solver = maker(suite, train_data, instances)
    successes = 0

    for (i, instance_id) in enumerate(test_data.run_lists):
        logger.info("simulating run %i/%i on %s", i, len(test_data), instance_id)

        with suite.domain.task_from_path(instance_id) as instance:
            with borg.accounting() as accountant:
                answer = solver.start(instance).run_then_stop(budget)

            succeeded = suite.domain.is_final(instance, answer)

            logger.info(
                "%s %s on %s (%.2f CPU s)",
                maker.name,
                "succeeded" if succeeded else "failed",
                os.path.basename(instance),
                accountant.total.cpu_seconds,
                )

            if succeeded:
                successes += 1

    logger.info(
        "%s had %i successes over %i instances",
        maker.name,
        successes,
        len(test_data),
        )

    return [(run["category"], maker.name, instances, successes)]

@borg.annotations(
    out_path = ("results CSV output path"),
    runs = ("path to JSON runs specification", "positional", None, borg.util.load_json),
    repeats = ("number of times to repeat each run", "option", None, int),
    workers = ("submit jobs?", "option", "w"),
    local = ("workers are local?", "flag"),
    )
def main(out_path, runs, repeats = 4, workers = 0, local = False):
    """Simulate portfolio and solver behavior."""

    logger.info("simulating %i runs", len(runs) * repeats)

    get_run_data = borg.util.memoize(borg.storage.RunData.from_bundle)

    #numpy.random.seed(42)
    random_state = numpy.random

    def yield_jobs():
        for run in runs:
            all_data = get_run_data(run["bundle"])
            #validation = sklearn.cross_validation.KFold(len(train_data), repeats, indices = False)
            validation = sklearn.cross_validation.ShuffleSplit(len(all_data), 32, test_fraction = 0.5, indices = False, random_state = random_state)
            maker = PortfolioMaker(run["portfolio_name"])

            for (train_mask, test_mask) in validation:
                import numpy
                for instances in map(int, map(round, numpy.r_[10.0:150.0:32j])):
                #for instances in [60]:
                    yield (simulate_run, [run, maker, all_data, train_mask, test_mask, instances])
                #break # XXX

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["category", "solver", "instances", "successes"])

        jobs = list(yield_jobs())
        if workers == "auto":
            workers = min(256, len(jobs))
        else:
            workers = int(workers)

        for (_, rows) in condor.do(jobs, workers, local):
            writer.writerows(rows)

            out_file.flush()

if __name__ == "__main__":
    borg.script(main)

