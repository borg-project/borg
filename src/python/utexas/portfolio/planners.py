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

    def __init__(self, discount):
        """
        Initialize.
        """

        self.discount = discount

    def select(self, predicted, utilities):
        """
        Select an action given the probabilities of outcomes.
        """

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

#         log.detail("predicted probabilities of success:\n%s", lines)

        (selected, _) = \
            max(
                predicted.iteritems(),
                key = lambda (a, ps): sum(p*u*self.discount**a.cutoff.as_s for (p, u) in zip(ps, utilities)),
                )

        return selected

class SoftMyopicActionPlanner(ActionPlanner):
    """
    Probabilistic greedy action selection.
    """

    def __init__(self, world, discount, temperature = 1.0):
        """
        Initialize.
        """

        self.world       = world
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

