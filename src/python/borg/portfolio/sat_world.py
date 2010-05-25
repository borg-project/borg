"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log            import get_logger
from borg.portfolio.world import (
    AbstractAction,
    AbstractOutcome,
    AbstractTrainer,
    )

log = get_logger(__name__)

class SAT_WorldAction(AbstractAction):
    """
    An action in the world.
    """

    def __init__(self, solver, budget):
        """
        Initialize.
        """

        self._solver = solver
        self._budget = budget

    @property
    def description(self):
        """
        A human-readable description of this action.
        """

        return "%s_%ims" % (self.solver.name, int(self.cost * 1000))

    @property
    def cost(self):
        """
        The typical cost of taking this action.
        """

        return self._budget.as_s

    @property
    def budget(self):
        """
        The time-valued cost of taking this action.
        """

        return self._budget

    @property
    def outcomes(self):
        """
        The possible outcomes of this action.
        """

        return SAT_WorldOutcome.BY_INDEX

    @property
    def solver(self):
        """
        The solver associated with this SAT action.
        """

        return self._solver

class SAT_WorldOutcome(AbstractOutcome):
    """
    An outcome of an action in the world.
    """

    def __init__(self, n, utility):
        """
        Initialize.
        """

        self.n        = n
        self._utility = utility

    def __str__(self):
        """
        Return a human-readable description of this outcome.
        """

        return str(self._utility)

    @property
    def utility(self):
        """
        The utility of this outcome.
        """

        return self._utility

    @staticmethod
    def from_result(attempt):
        """
        Return an outcome from a solver attempt.
        """

        if attempt.answer is None:
            return SAT_WorldOutcome.UNSOLVED
        else:
            return SAT_WorldOutcome.from_bool(attempt.answer.satisfiable)

    @staticmethod
    def from_bool(bool):
        """
        Return an outcome from True, False, or None.
        """

        return SAT_WorldOutcome.BY_VALUE[bool]

# outcome constants
SAT_WorldOutcome.SOLVED   = SAT_WorldOutcome(0, 1.0)
SAT_WorldOutcome.UNSOLVED = SAT_WorldOutcome(1, 0.0)
SAT_WorldOutcome.BY_VALUE = {
    True:  SAT_WorldOutcome.SOLVED,
    False: SAT_WorldOutcome.SOLVED,
    None:  SAT_WorldOutcome.UNSOLVED,
    }
SAT_WorldOutcome.BY_INDEX = [
    SAT_WorldOutcome.SOLVED,
    SAT_WorldOutcome.UNSOLVED,
    ]

