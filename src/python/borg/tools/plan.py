"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

@borg.annotations(
    out_path = ("plan output path"),
    suite_path = ("path to the solvers suite", "positional", None, os.path.abspath),
    bundle_path = ("path to runs bundle", "positional", None, os.path.abspath),
    suffix = ("runs data file suffix", "option"),
    )
def main(
    out_path,
    suite_path,
    bundle_path,
    suffix = ".runs.csv",
    ):
    """Evaluate portfolio performance under a specified model."""

    borg.get_logger("borg.models", level = "WARNING")

    logger.info("loading run data")

    suite = borg.load_solvers(suite_path)
    solver_names = list(suite.solvers)
    #paths = list(borg.util.files_under(instances_root, suite.domain.extensions))
    run_data = borg.RunData.from_bundle(bundle_path)

    logger.info("computing a plan over %i instances", len(run_data))

    model = \
        borg.models.mean_posterior(
            borg.models.MulSampler(),
            run_data.solver_names,
            run_data,
            bins = 30,
            chains = 1,
            samples_per_chain = 1,
            )
    planner = borg.planners.KnapsackPlanner()
    plan = planner.plan(model.log_survival[..., :-1], model.log_weights)

    logger.info("writing plan to %s", out_path)

    with open(out_path, "w") as out_file:
        out_csv = csv.writer(out_file)

        out_csv.writerow(["solver", "start", "end"])

        t = 0

        for (s, d) in plan:
            out_csv.writerow(map(str, [solver_names[s], t, t + d + 1]))

            t += d + 1

if __name__ == "__main__":
    borg.script(main)

