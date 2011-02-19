"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.get_features import main

    plac.call(main)

import re
import os.path
import csv
import cargo

logger = cargo.get_logger(__name__, default_level = "INFO")

def get_features_for(cnf_path):
    """Obtain features of a CNF."""

    home = "/scratch/cluster/bsilvert/sat-competition-2011/solvers/features1s"
    command = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/run-1.4/run",
        "-k",
        os.path.join(home, "features1s"),
        cnf_path,
        ]

    def set_library_path():
        os.environ["LD_LIBRARY_PATH"] = "{0}:{1}".format(home, os.environ["LD_LIBRARY_PATH"])

    (stdout, stderr, code) = cargo.call_capturing(command, preexec_fn = set_library_path)

    match = re.search(r"^\[run\] time:[ \t]*(\d+.\d+) seconds$", stderr, re.M)
    (cost,) = map(float, match.groups())

    (names, values) = [l.split(",") for l in stdout.splitlines()[-2:]]
    values = map(float, values)

    logger.info("collected features for %s in %.2f s", cnf_path, cost)

    return (["cost"] + names, [cost] + values)

@plac.annotations(
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(tasks_root, workers = 0):
    """Collect task features."""

    cargo.enable_default_logging()

    def yield_runs():
        paths = list(cargo.files_under(tasks_root, ["*.cnf"]))

        for path in paths:
            yield (get_features_for, [path])

    def collect_run((_, arguments), rows):
        (cnf_path,) = arguments
        csv_path = cnf_path + ".features.csv"

        with open(csv_path, "w") as csv_file:
            csv.writer(csv_file).writerows(rows)

    cargo.distribute_or_labor(yield_runs(), workers, collect_run)

