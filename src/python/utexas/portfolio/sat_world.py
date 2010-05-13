"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

# import numpy

# from utexas.sat.solvers      import get_random_seed
from utexas.portfolio.world  import (
    Action,
    Outcome,
    )

# class SAT_WorldTask(Task):
#     """
#     A task in the world.
#     """

#     def __init__(self, path, name = None):
#         """
#         Initialize.

#         @param task: SAT task description.
#         """

#         self.path = path
#         self.name = name

#     def __str__(self):
#         """
#         Return a human-readable description of this task.
#         """

#         if self.name is None:
#             return self.path
#         else:
#             return self.name

class SAT_WorldAction(Action):
    """
    An action in the world.
    """

    def __init__(self, solver, budget):
        """
        Initialize.
        """

        self._solver = solver
        self._cost   = budget
#         self._outcomes = SAT_WorldOutcome.BY_INDEX

    @property
    def description(self):
        """
        A human-readable description of this action.
        """

        return "%s_%ims" % (self.solver.name, int(self.cost.as_s * 1000))

    @property
    def solver(self):
        """
        The solver associated with this SAT action.
        """

        return self._solver

    @property
    def cost(self):
        """
        The typical cost of taking this action.
        """

        return self._cost

#     def take(self, task, random = numpy.random):
#         """
#         Take the action.
#         """

#         if self.solver.seeded:
#             seed = get_random_seed(random)
#         else:
#             seed = None

#         result  = self.solver.solve(task.path, self.cutoff, seed = seed)
#         outcome = SAT_WorldOutcome.from_result(result)

#         return (outcome, result)

class SAT_WorldOutcome(Outcome):
    """
    An outcome of an action in the world.
    """

    def __init__(self, n, utility):
        """
        Initialize.
        """

        self.n       = n
        self.utility = utility

    def __str__(self):
        """
        Return a human-readable description of this outcome.
        """

        return str(self.utility)

    @staticmethod
    def from_result(result):
        """
        Return an outcome from a solver result.
        """

        return SAT_WorldOutcome.from_bool(result.satisfiable)

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

