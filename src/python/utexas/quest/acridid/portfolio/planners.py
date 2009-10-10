"""
utexas/papers/nips2009/planners.py

Model-based action planning.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc import (
    ABCMeta,
    abstractmethod)
from utexas.alog import DefaultLogger

log = DefaultLogger("utexas.papers.nips2009.planners")

class ActionPlanner(object):
    """
    Interface for action selection schemes.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def select(self, predicted):
        """
        Select an action given the probabilities of outcomes.
        """

        pass

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

    def select(self, predicted, actions, task_history):
        """
        Select an action given the probabilities of outcomes.
        """

        # convert to expectation
#         outcomes = numpy.minimum(numpy.sum(task_history, 0), 1)
#         center = numpy.max(outcomes * self.world.utilities)
#         recentered = numpy.maximum(self.world.utilities - center, 0.0)
#         expected = numpy.sum(predicted * recentered, 1)
        expected = numpy.sum(predicted * self.world.utilities, 1)
        selected = max(actions, key = lambda a: expected[a.n]*(self.discount**a.cutoff))

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

