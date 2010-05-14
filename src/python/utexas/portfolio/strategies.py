"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc         import abstractmethod
from cargo.log   import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

class SelectionStrategy(ABC):
    """
    Abstract base for selection strategies.
    """

    @abstractmethod
    def select(self, task, budget):
        """
        Select an action, yield it, and receive its outcome.
        """

class SequenceSelectionStrategy(SelectionStrategy):
    """
    A strategy the follows an iterable sequence.
    """

    def __init__(self, actions):
        """
        Initialize.
        """

        self.action_sequence = iter(actions)

    def select(self, task, budget):
        """
        Select an action, yield it, and receive its outcome.
        """

        selected = self.action_sequence.next()

        if selected.cost > budget:
            yield None
        else:
            yield selected

class FixedSelectionStrategy(SequenceSelectionStrategy):
    """
    A strategy that repeats a fixed action.
    """

    def __init__(self, action):
        """
        Initialize.
        """

        from itertools import repeat

        SequenceSelectionStrategy.__init__(self, repeat(action))

class ModelingSelectionStrategy(SelectionStrategy):
    """
    A strategy that employs a model of its actions.
    """

    def __init__(self, model, planner):
        """
        Initialize.
        """

        self.model   = model
        self.planner = planner
        self.history = []

    def select(self, task, budget):
        """
        Select an action, yield it, and receive its outcome.
        """

        # predict, then make a selection
        predicted = self.model.predict(task, self.history)
        selected  = self.planner.select(predicted, budget)
        outcome   = yield selected

        # remember its result
        if outcome is not None:
            self.history.append((task, selected, outcome))

