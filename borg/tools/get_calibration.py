"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import borg

logger = borg.get_logger(__name__, default_level = "DETAIL")

def read_median_cost(path):
    runs_data = numpy.recfromcsv(path, usemask = True)

    return numpy.median(runs_data["cost"])

@borg.annotations()
def main(local_path, train_path):
    """Compute the machine speed calibration ratio."""

    local_median = read_median_cost(local_path)
    train_median = read_median_cost(train_path)

    logger.info("local median run time is %.2f CPU seconds", local_median)
    logger.info("model median run time is %.2f CPU seconds", train_median)
    logger.info("local speed ratio is thus %f", local_median / train_median)

if __name__ == "__main__":
    borg.script(main)

