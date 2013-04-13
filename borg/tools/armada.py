"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.armada import main

    plac.call(main)

import sys
import random
import logging
import cPickle as pickle
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    in_path = ("path to instance"),
    seed = ("PRNG seed", "option", None, int),
    workers = ("units of execution", "option", None, int),
    )
def main(in_path, seed = 42, budget = 2e6, workers = 1):
    """Solve a problem instance with an armada."""

    # general setup
    numpy.random.seed(seed)
    random.seed(numpy.random.randint(2**31))

    # parse the instance
    with open(in_path) as in_file:
        cnf = borg.domains.sat.dimacs.parse_cnf(in_file)

    # generate splits
    nbits = int(math.ceil(math.log(workers)))
    bits = sorted(xrange(cnf.N), key = lambda: random.random())[:nbits]

    for bit in bits:
        for polarity in [-1, 1]:
            clauses = cnf.clauses + [[polarity * (bit + 1)]]
            subcnf = borg.domains.sat.dimacs.DIMACS_GraphFile([], clauses, cnf.N)

