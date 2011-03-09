"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.calibrate import main

    plac.call(main)

import re
import os.path
import csv
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "DETAIL")

def collect_calibration_datum():
    """Run a known solver on a known instance."""

    command = [
        "/usr/bin/time",
        "-f", "%U",
        os.path.join(borg.defaults.solvers_root, "gnovelty+2/gnovelty+2"),
        os.path.join(borg.defaults.solvers_root, "calibration/unif-k7-r89-v75-c6675-S342542912-045.cnf"),
        "42",
        ]

    logger.detail("running %s", command)

    (stdout, stderr, code) = cargo.call_capturing(command)

    return float(stderr.splitlines()[-1])

@plac.annotations(
    runs = ("number of runs", "option", "r", int),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(out_path, runs = 9, workers = 0):
    """Collect solver running-time data."""

    cargo.enable_default_logging()

    def yield_runs():
        for _ in xrange(runs):
            yield (collect_calibration_datum, [])

    data = []

    def collect_run(_, datum):
        logger.info("run completed in %.2f CPU seconds", datum)

        data.append(datum)

    cargo.distribute_or_labor(yield_runs(), workers, collect_run)

    median = sorted(data)[len(data) / 2]

    logger.info("median run time is %.2f CPU seconds", median)

    with open(out_path, "w") as out_file:
        out_file.write("{0}\n".format(median))

