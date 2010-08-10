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

        from itertools import izip

        best_action      = None
        best_expectation = None

        for (i, action) in enumerate(self.actions):
            if action.cost <= budget:
                e  = sum(p * o.utility for (p, o) in izip(predicted[i], action.outcomes))
                e *= self.discount**action.cost

                if best_action is None or best_expectation < e:
                    best_action      = action
                    best_expectation = e

        return best_action

    @staticmethod
    def build(request, trainer, model):
        """
        Build a sequence strategy as requested.
        """

        return HardMyopicPlanner(model.actions, request["discount"])

