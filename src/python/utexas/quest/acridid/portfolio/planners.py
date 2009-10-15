"""
utexas/papers/nips2009/planners.py

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

class ActionPlanner(ABC):
    """
    Interface for action selection schemes.
    """

    @abstractmethod
    def select(self, predicted, actions):
        """
        Select an action given the probabilities of outcomes.
        """

        assert False

class HardMyopicActionPlanner(ActionPlanner):
    """
    Deterministic greedy action selection.
    """

    def __init__(self, world, discount):
        """
        Initialize.
        """

        self.world = world
        self.discount = discount

    def select(self, predicted, actions):
        """
        Select an action given the probabilities of outcomes.
        """

        # convert to expectation
#         outcomes = numpy.minimum(numpy.sum(task_history, 0), 1)
#         center = numpy.max(outcomes * self.world.utilities)
#         recentered = numpy.maximum(self.world.utilities - center, 0.0)
#         expected = numpy.sum(predicted * recentered, 1)
        expected = numpy.sum(predicted * self.world.utilities, 1)
        selected = max(actions, key = lambda a: expected[a.n]*(self.discount**a.cutoff.as_s))

        return selected

class SoftMyopicActionPlanner(ActionPlanner):
    """
    Probabilistic greedy action selection.
    """

    def __init__(self, world, discount):
        """
        Initialize.
        """

        self.world = world
        self.discount = discount

    def select(self, predicted, actions):
        """
        Select an action given the probabilities of outcomes.
        """

        # convert to expectation
        expected = numpy.sum(predicted * self.world.utilities, 1)
        probabilities = numpy.fromiter((expected[a.n]*(self.discount**a.cutoff) for a in actions), numpy.double)
        probabilities /= numpy.sum(probabilities)
        ((naction,),) = numpy.nonzero(numpy.random.multinomial(1, probabilities))

        return actions[naction]

