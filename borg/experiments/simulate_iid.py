"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import numpy
import sklearn
import condor
import borg
import borg.experiments.simulate_runs

logger = borg.get_logger(__name__, default_level = "INFO")

def simulate_run(run, maker, all_data, train_mask, test_mask, instances, independent, mixture):
    """Simulate portfolio execution on a train/test split."""

    train_data = all_data.masked(train_mask)
    test_data = all_data.masked(test_mask)

    if instances is not None:
        ids = sorted(train_data.run_lists, key = lambda _: numpy.random.rand())[:instances]
        train_data = train_data.filter(*ids)

    if independent:
        train_data = train_data.collect_independent(mixture).only_nonempty()
    else:
        train_data = train_data.collect_systematic(mixture).only_nonempty()

    budget = test_data.common_budget
    #budget = test_data.common_budget  / 2 # XXX
    suite = borg.fake.FakeSuite(test_data)

    if maker.subname == "preplanning-dir":
        model_kwargs = {"K": 64}

        if "set_alpha" in maker.variants:
            model_kwargs["alpha"] = 1e-2
    else:
        model_kwargs = {}

    solver = maker(suite, train_data, model_kwargs = model_kwargs)
    successes = []

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
                successes.append(accountant.total.cpu_seconds)

    logger.info(
        "%s had %i successes over %i instances",
        maker.name,
        len(successes),
        len(test_data),
        )

    description = "{0} ({1})".format(mixture, "Sep." if independent else "Sys.")

    return (
        description,
        maker.name,
        instances,
        len(successes),
        numpy.mean(successes),
        numpy.median(successes),
        )

@borg.annotations(
    out_path = ("results CSV output path"),
    runs = ("path to JSON runs specification", "positional", None, borg.util.load_json),
    repeats = ("number of times to repeat each run", "option", None, int),
    workers = ("submit jobs?", "option", "w"),
    local = ("workers are local?", "flag"),
    )
def main(out_path, runs, repeats = 128, workers = 0, local = False):
    """Simulate portfolio and solver behavior."""

    logger.info("simulating %i runs", len(runs))

    get_run_data = borg.util.memoize(borg.storage.RunData.from_bundle)

    def yield_jobs():
        for run in runs:
            all_data = get_run_data(run["bundle"])
            validation = sklearn.cross_validation.ShuffleSplit(len(all_data), repeats, test_fraction = 0.2, indices = False)

            if run["portfolio_name"] == "-":
                makers = map(borg.experiments.simulate_runs.SolverMaker, all_data.solver_names)
            else:
                makers = [borg.experiments.simulate_runs.PortfolioMaker(run["portfolio_name"])]

            max_instances = len(all_data) * 0.8

            for (train_mask, test_mask) in validation:
                for instances in map(int, map(round, numpy.r_[10.0:max_instances:32j])):
                    for maker in makers:
                        yield (
                            simulate_run,
                            [
                                run,
                                maker,
                                all_data,
                                train_mask,
                                test_mask,
                                instances,
                                run["independent"],
                                run["mixture"],
                                ],
                            )

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["description", "solver", "instances", "successes", "mean_time", "median_time"])

        for (_, row) in condor.do(yield_jobs(), workers, local):
            writer.writerow(row)

            out_file.flush()

if __name__ == "__main__":
    borg.script(main)

