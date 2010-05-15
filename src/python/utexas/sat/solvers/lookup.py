"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.solvers.base import SAT_Solver

class SAT_LookupSolver(SAT_Solver):
    """
    A solver which indirectly executes a named solver.
    """

    def __init__(self, name):
        """
        Initialize.
        """

        self.name = name

    def solve(self, task, budget, random, environment):
        """
        Look up the named solver and use it on the task.
        """

        from utexas.sat.solvers.base import SAT_WrappedResult

        inner_solver = environment.named_solvers[self.name]
        inner_result = inner_solver.solve(task, budget, random, environment)

        return SAT_WrappedResult(self, inner_result)

    def to_orm(self):
        """
        Return a database description of this solver.
        """

        return SAT_SolverRow(name = self.name)

