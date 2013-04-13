"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import csv
import itertools
import numpy
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def run_experiment(run_data, planner_name, B):
    if planner_name == "knapsack":
        planner = borg.planners.KnapsackPlanner()
    elif planner_name == "streeter":
        planner = borg.planners.StreeterPlanner()
    elif planner_name == "bellman":
        planner = borg.planners.BellmanPlanner()
    else:
        raise ValueError("unrecognized planner name \"{0}\"".format(planner_name))

    suite = borg.fake.FakeSuite(run_data)
    portfolio = borg.portfolios.PreplanningPortfolio(suite, run_data, B = B, planner = planner)

    def yield_rows():
        for instance_id in run_data.run_lists:
            with suite.domain.task_from_path(instance_id) as instance:
                budget = borg.Cost(cpu_seconds = run_data.common_budget)
                answer = portfolio(instance, suite, budget)
                succeeded = suite.domain.is_final(instance, answer)

                yield 1.0 if succeeded else 0.0

    rate = numpy.mean(list(yield_rows()))

    logger.info("success rate with B = %i is %f", B, rate)

    return [planner_name, B, rate]

@borg.annotations(
    out_path = ("results output path"),
    bundle = ("path to pre-recorded runs"),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, bundle, workers = 0, local = False):
    """Evaluate the mixture model(s) over a range of component counts."""

    def yield_jobs():
        run_data = borg.storage.RunData.from_bundle(bundle)
        planner_names = ["knapsack", "streeter", "bellman"]
        bin_counts = xrange(1, 121)
        replications = xrange(16)
        experiments = itertools.product(planner_names, bin_counts, replications)

        for (planner_name, bin_count, _) in experiments:
            if planner_name != "bellman" or bin_count <= 5:
                yield (run_experiment, [run_data, planner_name, bin_count])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["planner", "bins", "rate"])

        for (_, row) in condor.do(yield_jobs(), workers, local):
            writer.writerow(row)

if __name__ == "__main__":
    borg.script(main)

