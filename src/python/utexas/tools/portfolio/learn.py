# vim: set fileencoding=UTF-8 :
"""
utexas/tools/portfolio/train.py

Solve a task using a portfolio.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.solve import main

    raise SystemExit(main())

import re
import json
import logging
import numpy

from os.path            import (
    join,
    dirname,
    )
from copy               import copy
from tempfile           import NamedTemporaryFile
from cargo.io           import expandpath
from cargo.log          import get_logger
from cargo.json         import follows
from cargo.flags        import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.temporal     import TimeDelta
from utexas.sat.solvers import get_named_solvers

log = get_logger(__name__, level = logging.NOTE)

module_flags = \
    Flags(
        "Script Options",
        Flag(
            "-t",
            "--training",
            default = "training.json",
            metavar = "FILE",
            help    = "load training data from FILE [%default]",
            ),
        Flag(
            "-o",
            "--output",
            default = "portfolio.json",
            metavar = "FILE",
            help    = "write configuration to FILE [%default]",
            ),
        )

# portfolio configuration information:
# - model name
# - model configuration
# - planner name
# - planner configuration

def smooth_mult(mixture):
    """
    Apply a smoothing term to the multinomial mixture components.
    """

    log.info("heuristically smoothing mult mixture")

    epsilon = 1e-3

    for m in xrange(mixture.ndomains):
        for k in xrange(mixture.ncomponents):
            beta                      = mixture.components[m, k].beta + epsilon
            beta                     /= numpy.sum(beta)
            mixture.components[m, k]  = Multinomial(beta)

def make_mult_mixture_model(world, training, ncomponents):
    """
    Return a new multinomial mixture strategy for evaluation.
    """

    log.info("building multinomial mixture model")

    model  = \
        MultinomialMixtureActionModel(
            world,
            training,
            ExpectationMaximizationMixtureEstimator(
                [[MultinomialEstimator()] * ncomponents] * world.nactions,
                ),
            )

    smooth_mult(model.mixture)

    return model

def train_portfolio(tasks, solvers, cutoffs):
    """
    Return trained portfolio.
    """

    model = make_mult_mixture_model(world, train_history, 4)

"""
    # build the strategy
    if isinstance(s_name, SAT_WorldAction):
        # fixed strategy
        strategy = FixedSelectionStrategy(s_name)
    elif s_name == "model":
        # model-based strategy; build the model
        if model_name == "dcm":
            model = make_dcm_mixture_model(world, train_history, K)
        elif model_name == "random":
            model = RandomActionModel(world)
        else:
            raise ValueError()

        # build the planner
        if planner_name == "hard":
            planner = HardMyopicActionPlanner(world, discount)
        elif planner_name == "soft":
            planner = SoftMyopicActionPlanner(world, discount)
        else:
            raise ValueError()

        strategy = ModelingSelectionStrategy(world, model, planner)
    else:
        raise ValueError()
"""

@with_flags_parsed(
    usage = "usage: %prog [options]",
    )
def main(positional):
    """
    Main.
    """

    # basic flag handling
    flags = module_flags.given

    if flags.verbose:
        get_logger("utexas.tools.sat.run_solvers").setLevel(logging.NOTSET)
        get_logger("cargo.unix.accounting").setLevel(logging.DEBUG)
        get_logger("utexas.sat.solvers").setLevel(logging.DEBUG)

