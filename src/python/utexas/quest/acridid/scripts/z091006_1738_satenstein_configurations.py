"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.quest.acridid.scripts.z091006_1738_satenstein_configurations import main

    raise SystemExit(main())

import os
import os.path
import sys
import logging
import numpy

from uuid import UUID
from datetime import timedelta
from collections import namedtuple
from cargo.ai.sat.solvers import (
    SATensteinSolver,
    )
from cargo.log import get_logger
from cargo.flags import parse_given
from utexas.quest.acridid.core import (
    AcrididSession,
    acridid_connect,
    SAT_ConfigurationSet,
    SATensteinConfiguration,
    )

log                   = get_logger(__name__, level = None)
Parameter             = namedtuple("Parameter", ["domain", "default"])
SATENSTEIN_PARAMETERS = {
    "adaptive":                Parameter([0, 1], 1),
    "adaptivenoisescheme":     Parameter([1, 2], 1),
    "adaptiveprom":            Parameter([0, 1], 0),
    "adaptpromwalkprob":       Parameter([0, 1], 0),
    "adaptwalkprob":           Parameter([0, 1], 0),
    "alpha":                   Parameter([1.3], 1.3),
    "c":                       Parameter([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], 0.01),
    "clausepen":               Parameter([0, 1], 1),
    "decreasingvariable":      Parameter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 3),
    "dp":                      Parameter([0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.20], 0.05),
    "heuristic":               Parameter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 2),
    "maxinc":                  Parameter([10], 10),
    "novnoise":                Parameter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 , 0.8], 0.5),
    "performalternatenovelty": Parameter([0, 1], 1),
    "performrandomwalk":       Parameter([0, 1], 0),
    "pflat":                   Parameter([0.15], 0.15),
    "phi":                     Parameter([3, 4, 5, 6, 7, 8, 9, 10], 5),
    "promdp":                  Parameter([0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.20], 0.05),
    "promisinglist":           Parameter([0, 1], 1),
    "promphi":                 Parameter([3, 4, 5, 6, 7, 8, 9, 10], 5),
    "promnovnoise":            Parameter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 , 0.8], 0.5),
    "promtheta":               Parameter([3, 4, 5, 6, 7, 8, 9, 10], 6),
    "promwp":                  Parameter([0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.20], 0.01),
    "ps":                      Parameter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0.4),
    "randomwalk":              Parameter([1, 3, 4, 5], 1),
    "rdp":                     Parameter([0.01, 0.03, 0.05, 0.07, 0.1, 0.15], 0.05),
    "rfp":                     Parameter([0.01, 0.03, 0.05, 0.07, 0.1, 0.15], 0.01),
    "rho":                     Parameter([0.8], 0.8),
    "rwp":                     Parameter([0.01, 0.03, 0.05, 0.07, 0.1, 0.15], 0.01),
    "rwpwalk":                 Parameter([0.01, 0.03, 0.05, 0.07, 0.1, 0.15], 0.05),
    "s":                       Parameter([0.1, 0.01, 0.001], 0.01),
    "sapsthresh":              Parameter([-0.1], -0.1),
    "scoringmeasure":          Parameter([1], 1),
    "selectclause":            Parameter([1, 7], 1),
    "singleclause":            Parameter([1], 1),
    "smoothingscheme":         Parameter([1, 2], 1),
    "tabu":                    Parameter([1, 3, 5, 7, 10, 15, 20], 1),
    "tabusearch":              Parameter([0, 1], 0),
    "theta":                   Parameter([3, 4, 5, 6, 7, 8, 9, 10], 6),
    "tiebreaking":             Parameter([1, 2, 3, 4], 1),
    "updateschemepromlist":    Parameter([1, 2, 3], 3),
    "varinfalse":              Parameter([1], 1),
    "wp":                      Parameter([0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.20], 0.01),
    "wpwalk":                  Parameter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 0.5),
    }

def pick_one(array):
    """
    Pick a random element from an array.
    """

    return array[numpy.random.randint(len(array))]

def main():
    """
    Application body.
    """

    positional = parse_given()

    AcrididSession.configure(bind = acridid_connect())

    session           = AcrididSession()
    configuration_set = \
        session.merge(
            SAT_ConfigurationSet(
                uuid = UUID("b5bad358baa04841a735650016591b19"),
                name = "randomly generated SATenstein configurations (first attempt)",
                ),
            )

    for i in xrange(16):
        parameters    = dict((n, pick_one(p.domain)) for (n, p) in SATENSTEIN_PARAMETERS.iteritems())
        configuration = session.merge(SATensteinConfiguration.from_parameters(parameters, configuration_set))

#     solver     = SATensteinSolver(parameters)

#     print result

    session.commit()

