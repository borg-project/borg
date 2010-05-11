"""
utexas/portfolio/sat_world.py

The world of SAT.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.log               import get_logger
from utexas.sat.solvers      import get_random_seed
from utexas.portfolio.world  import (
    Task,
    World,
    Action,
    Outcome,
    )

log = get_logger(__name__)

# module_flags = \
#     Flags(
#         "SAT Data Storage",
#         Flag(
#             "--sat-world-cache",
#             default = "sqlite:///:memory:",
#             metavar = "DATABASE",
#             help    = "use research DATABASE by default [%default]",
#             ),
#         )

class SAT_WorldTask(Task):
    """
    A task in the world.
    """

    def __init__(self, path, name = None):
        """
        Initialize.

        @param task: SAT task description.
        """

        self.path = path
        self.name = name

    def __str__(self):
        """
        Return a human-readable description of this task.
        """

        if self.name is None:
            return self.path
        else:
            return self.name

    def __json__(self):
        """
        Make JSONable.
        """

        return (self.path, self.name)

class SAT_WorldAction(Action):
    """
    An action in the world.
    """

    def __init__(self, solver, cutoff):
        """
        Initialize.
        """

        self.solver   = solver
        self.cutoff   = cutoff
        self.cost     = cutoff
        self.outcomes = SAT_WorldOutcome.BY_INDEX

    def __str__(self):
        """
        Return a human-readable description of this action.
        """

        return "%s_%ims" % (self.solver.name, int(self.cutoff.as_s * 1000))

    def __json__(self):
        """
        Make JSONable.
        """

        return (self.solver.name, self.cutoff.as_s)

    def take(self, task, random = numpy.random):
        """
        Take the action.
        """

        if self.solver.seeded:
            seed = get_random_seed(random)
        else:
            seed = None

        result  = self.solver.solve(task.path, self.cutoff, seed = seed)
        outcome = SAT_WorldOutcome.from_result(result)

        return (outcome, result)

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

    def __json__(self):
        """
        Make JSONable.
        """

        return self.n

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

