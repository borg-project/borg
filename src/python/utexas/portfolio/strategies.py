"""
utexas/portfolio/strategies.py

General selection strategies.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc import abstractmethod
from cargo.log import get_logger
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

    def __init__(self, world, model, planner):
        """
        Initialize.
        """

        self.world   = world
        self.model   = model
        self.planner = planner
        self.history = []

    def select(self, task, actions):
        """
        Select an action, yield it, and receive its outcome.
        """

        # predict, then make a selection
        predicted = self.model.predict(task, self.history)

        # FIXME don't do this debugging work always
#         from collections import defaultdict
#         from utexas.portfolio.sat_world import SAT_Outcome
#         map = defaultdict(list)

#         for (n, p) in enumerate(predicted):
#             a = self.world.actions[n]

#             map[a.solver.name].append((a.cutoff, p[SAT_Outcome.SOLVED.n]))

#         rows = []

#         for (s, m) in map.items():
#             rows.append((s, [p for (c, p) in sorted(m, key = lambda (c, p): c)]))

#         rows  = sorted(rows, key = lambda (s, r): s)
#         lines = "\n".join("% 32s: %s" % (s, " ".join("%.2f" % p for p in r)) for (s, r) in rows)

#         log.debug("predicted probabilities of success:\n%s", lines)

        # ...
        action    = self.planner.select(predicted, actions)
        outcome   = yield action

        # remember its result
        self.history.append((task, action, outcome))

