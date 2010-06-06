"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log               import get_logger
from borg.portfolio.planners import AbstractPlanner
from borg.portfolio._bellman import compute_bellman_utility

log = get_logger(__name__)

def compute_bellman_plan(model, horizon, budget, discount = 1.0):
    """
    Compute the Bellman-optimal plan.
    """

    import numpy

    dimensions = (len(model.actions), max(len(a.outcomes) for a in model.actions))
    history    = numpy.zeros(dimensions, numpy.uint)

    (_, plan) = compute_bellman_utility(model, horizon, budget, discount, history)

    return plan

