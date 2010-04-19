"""
utexas/portfolio/strategies.py

General selection strategies.

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
    def select(self, task):
        """
        Select an action, yield it, and receive its outcome.
        """

        pass

class ActionModel(ABC):
    """
    A model of action outcomes.
    """

    @abstractmethod
    def predict(self, task, history, out = None):
        """
        Return the predicted probability of each outcome given history.

        @return: numpy.ndarray(noutcomes, numpy.float); sums to one.
        """

        pass

class FixedSelectionStrategy(SelectionStrategy):
    """
    A strategy that repeats a fixed action.
    """

    name = "fixed"

    def __init__(self, action):
        """
        Initialize.
        """

        self.action = action

    def select(self, task, actions):
        """
        Select an action, yield it, and receive its outcome.
        """

        if self.action in actions:
            yield self.action
        else:
            yield None

class ModelingSelectionStrategy(SelectionStrategy):
    """
    A strategy that employs a model of its actions.
    """

    name = "modeling"

    def __init__(self, model, planner, actions):
        """
        Initialize.
        """

        self.model   = model
        self.planner = planner
        self.actions = actions
        self.history = []

        from utexas.portfolio.sat_world import SAT_WorldOutcome
        self.utilities = numpy.array([o.utility for o in SAT_WorldOutcome.BY_INDEX])

    def select(self, task, budget):
        """
        Select an action, yield it, and receive its outcome.
        """

        # predict, then make a selection
        if budget is None:
            feasible = self.actions
        else:
            feasible  = [a for a in self.actions if a.cutoff <= budget]

        predicted = self.model.predict(task, self.history, feasible)
        selected  = self.planner.select(predicted, self.utilities) # FIXME
        outcome   = yield selected

        # remember its result
        self.history.append((task, selected, outcome))

# assign strategy names
names = {
    FixedSelectionStrategy.name    : FixedSelectionStrategy,
    ModelingSelectionStrategy.name : ModelingSelectionStrategy,
    }

