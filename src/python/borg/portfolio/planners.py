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

def build_planner(request, trainer):
    """
    Build an action planner as requested.
    """

    builders = {
        "hard_myopic" : HardMyopicPlanner.build,
        "bellman"     : BellmanPlanner.build,
        }

    return builders[request["type"]](request, trainer)

class AbstractPlanner(ABC):
    """
    Interface for action selection schemes.
    """

    @abstractmethod
    def select(self, model, history, budget, random):
        """
        Select an action.
        """

class HardMyopicPlanner(AbstractPlanner):
    """
    Deterministic greedy action selection.
    """

    def __init__(self, discount):
        """
        Initialize.
        """

        self._discount = discount

    def select(self, model, history, budget, random):
        """
        Select an action.
        """

        # compute next-step predictions
        predicted = model.predict(history, random)

        # select an apparently-best action
        from itertools import izip

        best_action      = None
        best_expectation = None

        for (i, action) in enumerate(model.actions):
            if action.cost <= budget:
                e  = sum(p * o.utility for (p, o) in izip(predicted[i], action.outcomes))
                e *= self._discount**action.cost

                if best_action is None or best_expectation < e:
                    best_action      = action
                    best_expectation = e

        return best_action

    @staticmethod
    def build(request, trainer):
        """
        Build a sequence strategy as requested.
        """

        return HardMyopicPlanner(request["discount"])

class BellmanPlanner(AbstractPlanner):
    """
    Fixed-horizon optimal replanning.
    """

    def __init__(self, horizon, discount):
        """
        Initialize.
        """

        self._horizon  = horizon
        self._discount = discount

    def select(self, model, history, budget, random):
        """
        Select an action.
        """

        from borg.portfolio.bellman import compute_bellman_plan

        (utility, plan) = \
            compute_bellman_plan(
                model,
                self._horizon,
                budget,
                self._discount,
                history = history,
                )

        log.detail("computed Bellman plan: %s", " -> ".join(a.description for a in plan))

        return plan[0]

    @staticmethod
    def build(request, trainer):
        """
        Build a sequence strategy as requested.
        """

        return BellmanPlanner(request["horizon"], request["discount"])

