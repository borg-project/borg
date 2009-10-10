"""
utexas/papers/nips2009/strategies.py

General selection strategies.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc import (
    ABCMeta,
    abstractmethod)
from utexas.alog import DefaultLogger

log = DefaultLogger("utexas.papers.nips2009.strategies")

class SelectionStrategy(object):
    """
    Abstract base for selection strategies.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def select(self, task):
        """
        Select an action, yield it, and receive its outcome.
        """

        pass

class ActionHistory(object):
    """
    A history of action outcomes on tasks.
    """

    def __init__(self, world):
        """
        Initialize.
        """

        self.__world = world
        self.__counts = numpy.zeros((world.ntasks, world.nactions, world.noutcomes), numpy.uint)

    def add_outcome(self, task, action, outcome):
        """
        Add an outcome to the history.
        """

        self.__counts[task.n, action.n, outcome.n] += 1

    def get_positive_counts(self):
        """
        Get an array of counts for tasks with recorded outcomes.
        """

        return self.__counts[numpy.sum(numpy.sum(self.__counts, 1), 1) > 0]

    @staticmethod
    def sample(world, tasks, nrestarts, random = numpy.random):
        """
        Return a history of C{nrestarts} outcomes sampled from each of C{tasks}.
        """

        history = ActionHistory(world)

        for (nlesson, task) in enumerate(tasks):
            for action in world.actions:
                for nrestart in xrange(nrestarts):
                    outcome = world.sample_action(task, action)

                    history.add_outcome(task, action, outcome)

        return history

    # properties
    counts = property(lambda self: self.__counts)

class ActionModel(object):
    """
    A model of action outcomes.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, task, history, out = None):
        """
        Return the predicted probability of each outcome given history.
        """

        pass

class FixedSelectionStrategy(SelectionStrategy):
    """
    A strategy that repeats a fixed action.
    """

    def __init__(self, action):
        """
        Initialize.
        """

        self.action = action

    def select(self, task, actions):
        """
        Select an action, yield it, and receive its outcome.
        """

        assert self.action in actions

        yield self.action

class ModelingSelectionStrategy(SelectionStrategy):
    """
    A strategy that employs a model of its actions.
    """

    def __init__(self, world, model, planner):
        """
        Initialize.
        """

        self.world = world
        self.model = model
        self.planner = planner
        self.history = ActionHistory(world)

    def select(self, task, actions):
        """
        Select an action, yield it, and receive its outcome.
        """

        # predict, then make a selection
        predicted = self.model.predict(task, self.history)
        action = self.planner.select(predicted, actions, self.history.counts[task.n])
        outcome = yield action

        # remember its result
        self.history.add_outcome(task, action, outcome)

