"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac
import borg

if __name__ == "__main__":
    plac.call(borg.tools.plan.main)

import os.path
import csv
import cargo

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    out_path = ("plan output path"),
    suite_path = ("path to the solvers suite", "positional", None, os.path.abspath),
    instances_root = ("path to instances", "positional", None, os.path.abspath),
    suffix = ("runs data file suffix", "option"),
    threshold = ("plan interestingness threshold", "option", None, int),
    )
def main(
    out_path,
    suite_path,
    instances_root,
    suffix = ".runs.csv",
    threshold = 1,
    ):
    """Evaluate portfolio performance under a specified model."""

    cargo.enable_default_logging()
    cargo.get_logger("borg.models", level = "WARNING")

    logger.info("loading run data")

    suite = borg.load_solvers(suite_path)
    solver_names = list(suite.solvers)
    paths = list(cargo.files_under(instances_root, suite.domain.extensions))

    logger.info("computing a plan")

    run_data = borg.RunData.from_paths(solver_names, paths, suite.domain, suffix)
    model = borg.models.Mul_ModelFactory().fit(run_data.solver_names, run_data, B = 30, T = 1)
    planner = borg.planners.KnapsackMultiversePlanner()
    plan = planner.plan(model.log_survival[..., :-1], model.log_weights)

    logger.info("writing plan to %s", out_path)

    with open(out_path, "w") as out_file:
        out_csv = csv.writer(out_file)

        out_csv.writerow(["solver", "start", "end"])

        t = 0

        for (s, d) in plan:
            out_csv.writerow(map(str, [solver_names[s], t, t + d + 1]))

            t += d + 1

