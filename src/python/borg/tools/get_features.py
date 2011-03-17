"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.get_features import main

    plac.call(main)

import os.path
import csv
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(tasks_root, workers = 0):
    """Collect task features."""

    cargo.enable_default_logging()

    def yield_runs():
        #paths = list(cargo.files_under(tasks_root, ["*.cnf"]))
        paths = list(cargo.files_under(tasks_root, ["*.opb"])) # XXX

        for path in paths:
            #yield (borg.features.get_features_for, [path]) # XXX
            yield (borg.features.pb.path_compute_all, [path])

    def collect_run((_, arguments), (names, values)):
        (cnf_path,) = arguments
        csv_path = cnf_path + ".features.csv"

        with open(csv_path, "w") as csv_file:
            csv.writer(csv_file).writerow(names)
            csv.writer(csv_file).writerow(values)

    cargo.distribute_or_labor(yield_runs(), workers, collect_run)

