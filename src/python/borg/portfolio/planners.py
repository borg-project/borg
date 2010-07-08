"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import (
    ABCMeta,
    abstractmethod,
    )
from cargo.log   import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

def build_planner(request, trainer, model):
    """
    Build an action planner as requested.
    """

    builders = {
        "hard_myopic" : HardMyopicPlanner.build,
        "soft_myopic" : SoftMyopicPlanner.build,
        }

    return builders[request["type"]](request, trainer, model)

class AbstractPlanner(ABC):
    """
    Interface for action selection schemes.
    """

    @abstractmethod
    def select(self, predicted, budget, random):
        """
        Select an action given the probabilities of outcomes.
        """

class HardMyopicPlanner(AbstractPlanner):
    """
    Deterministic greedy action selection.
    """

    def __init__(self, actions, discount):
        """
        Initialize.
        """

        self.actions  = actions
        self.discount = discount

    def select(self, predicted, budget, random):
        """
        Select an action given the probabilities of outcomes.
        """

        best_action      = None
        best_expectation = None

        for (i, action) in enumerate(self.actions):
            if action.cost <= budget:
                e = predicted[i, 0] * self.discount**action.cost

                if best_action is None or best_expectation < e:
                    best_action      = action
                    best_expectation = e

        if best_action is not None:
            print "selected", best_action.solver.name, best_action.cost

        return best_action

    @staticmethod
    def build(request, trainer, model):
        """
        Build a sequence strategy as requested.
        """

        return HardMyopicPlanner(model.actions, request["discount"])

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

    def select(self, predicted, actions, random):
        """
        Select an action given the probabilities of outcomes.
        """

        # convert to expectation
        import numpy

        expected       = numpy.sum(predicted * self.world.utilities, 1)
        discounted     = numpy.fromiter((expected[a.n]*(self.discount**a.cutoff.as_s) for a in actions), numpy.double)
        probabilities  = numpy.exp(discounted / self.temperature)
        probabilities /= numpy.sum(probabilities)
        ((naction,),)  = numpy.nonzero(random.multinomial(1, probabilities))
        action         = actions[naction]

        log.detail("probabilities: %s", probabilities)
        log.detail("selected action %i (p = %.4f): %s", naction, probabilities[naction], action)

        return action

    @staticmethod
    def build(request, trainer, model):
        """
        Build a sequence strategy as requested.
        """

        return SoftMyopicPlanner(request["discount"], request["temperature"])

