"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.expand_runs import main

    plac.call(main)

import os.path
import csv
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    out_root = ("root of new runs files"),
    train_path = ("path to training data"),
    )
def main(out_root, train_path):
    """Expand training data into multiple files."""

    cargo.enable_default_logging()

    # read the runs
    (train_runs, _) = borg.storage.read_train_runs(train_path)

    logger.info("expanding runs from %i tasks", len(train_runs))

    # write them to a bunch of files
    for (task_path, runs_by_solver) in train_runs.items():
        runs_path = os.path.join(out_root, task_path) + ".rtd.csv.gz"

        logger.info("writing runs to %s", runs_path)

        runs_dir = os.path.dirname(runs_path)

        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

        with cargo.openz(runs_path, "w") as runs_file:
            writer = csv.writer(runs_file)

            writer.writerow(["solver", "seed", "budget", "cost", "answer"])

            for (solver, runs) in runs_by_solver.items():
                for (budget, cost, answer) in runs:
                    writer.writerow([solver, None, budget, cost, bool(answer)])

