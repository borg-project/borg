"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.get_calibration import main

    plac.call(main)

import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "DETAIL")

def read_median_cost(path):
    runs_data = numpy.recfromcsv(path, usemask = True)

    return numpy.median(runs_data["cost"])

@plac.annotations()
def main(local_path, train_path):
    """Compute the machine speed calibration ratio."""

    cargo.enable_default_logging()

    local_median = read_median_cost(local_path)
    train_median = read_median_cost(train_path)

    logger.info("local median run time is %.2f CPU seconds", local_median)
    logger.info("model median run time is %.2f CPU seconds", train_median)
    logger.info("local speed ratio is thus %f", local_median / train_median)

