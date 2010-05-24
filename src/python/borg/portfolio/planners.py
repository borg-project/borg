"""
utexas/portfolio/planners.py

Model-based action planning.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc import (
    ABCMeta,
    abstractmethod,
    )
from cargo.log import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

def build_planner(request, trainer):
    """
    Build an action planner as requested.
    """

    builders = {
        "hard_myopic" : HardMyopicPlanner.build,
        "soft_myopic" : SoftMyopicPlanner.build,
        }

    return builders[request["type"]](request, trainer)

class AbstractPlanner(ABC):
    """
    Interface for action selection schemes.
    """

    @abstractmethod
    def select(self, predicted, budget):
        """
        Select an action given the probabilities of outcomes.
        """

class HardMyopicPlanner(AbstractPlanner):
    """
    Deterministic greedy action selection.
    """

    def __init__(self, discount):
        """
        Initialize.
        """

        self.discount = discount

    def select(self, predicted, budget):
        """
        Select an action given the probabilities of outcomes.
        """

        feasible = [(a, ps) for (a, ps) in predicted.iteritems() if a.cost <= budget]

        if feasible:
            (selected, _) = \
                max(
                    feasible,
                    key = lambda (a, ps): sum(p*o.utility*self.discount**a.cost for (p, o) in zip(ps, a.outcomes)),
                    )

            return selected
        else:
            return None

    @staticmethod
    def build(request, trainer):
        """
        Build a sequence strategy as requested.
        """

        return HardMyopicPlanner(request["discount"])

class SoftMyopicPlanner(AbstractPlanner):
    """
    Probabilistic greedy action selection.
    """

    def __init__(self, discount, temperature = 1.0):
        """
        Initialize.
        """

        self.discount    = discount
        self.temperature = temperature

    def select(self, predicted, actions):
        """
        Select an action given the probabilities of outcomes.
        """

        # convert to expectation
        expected       = numpy.sum(predicted * self.world.utilities, 1)
        discounted     = numpy.fromiter((expected[a.n]*(self.discount**a.cutoff.as_s) for a in actions), numpy.double)
        probabilities  = numpy.exp(discounted / self.temperature)
        probabilities /= numpy.sum(probabilities)
        ((naction,),)  = numpy.nonzero(numpy.random.multinomial(1, probabilities))
        action         = actions[naction]

        log.detail("probabilities: %s", probabilities)
        log.detail("selected action %i (p = %.4f): %s", naction, probabilities[naction], action)

        return action

    @staticmethod
    def build(request, trainer):
        """
        Build a sequence strategy as requested.
        """

        return SoftMyopicPlanner(request["discount"], request["temperature"])

