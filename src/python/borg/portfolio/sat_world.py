"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc                  import abstractmethod
from cargo.sugar          import ABC
from borg.portfolio.world import (
    Action,
    Outcome,
    )

class Trainer(ABC):
    """
    Grant a portfolio access to training data.
    """

    @abstractmethod
    def build_actions(request):
        """
        Build a list of actions from a configuration request.
        """

    @abstractmethod
    def get_data(self, action):
        """
        Provide per-task {outcome: count} maps to the trainee.
        """

class SAT_Trainer(Trainer):
    """
    Grant a SAT portfolio access to training data.
    """

    def __init__(self, Session):
        """
        Initialize.
        """

        self._Session = Session

    def build_actions(self, request):
        """
        Build a list of actions from a configuration request.
        """

        # build the solvers and cutoffs
        from cargo.temporal import TimeDelta
        from borg.solvers   import LookupSolver

        solvers = [LookupSolver(s) for s in request["solvers"]]
        budgets = [TimeDelta(seconds = s) for s in request["budgets"]]

        # build the actions
        from itertools import product

        return [SAT_WorldAction(*a) for a in product(solvers, budgets)]

    def get_data(self, action):
        """
        Provide per-task {outcome: count} maps to the trainee.
        """

        with self._Session() as session:
            solver_row  = action.solver.get_row(session)
            answer_case = 
            expression  = \
                    select(
                        [
                            count(answer_case),
                            count(),
                            ],
                        )

class SAT_WorldAction(Action):
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

        return "%s_%ims" % (self.solver.name, int(self.cost.as_s * 1000))

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

class SAT_WorldOutcome(Outcome):
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

