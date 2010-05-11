"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.solvers import (
    SAT_Solver,
    SAT_BareResult,
    )

class SAT_PortfolioResult(SAT_BareResult):
    """
    Result of a portfolio solver.
    """

class SAT_PortfolioSolver(SAT_Solver):
    """
    Solve SAT instances with a portfolio.
    """

    def __init__(self, strategy):
        """
        Initialize.
        """

        self.strategy        = strategy
        self.max_invocations = 50

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Execute the solver and return its outcome, given an input path.
        """

        from numpy.random               import RandomState
        from utexas.portfolio.sat_world import SAT_WorldTask

        # get us a pseudorandom sequence
        if type(seed) is int:
            random = RandomState(seed)
        elif hasattr(seed, "rand"):
            random = seed
        else:
            raise ValueError("seed or PRNG required")

        # solve the instance
        (satisfiable, certificate) = \
            self._solve_on(
                SAT_WorldTask(input_path, input_path),
                cutoff,
                random,
                )

        return SAT_PortfolioResult(satisfiable, certificate)

    def _solve_on(self, task, cutoff, random):
        """
        Evaluate on a specific task.
        """

        from cargo.temporal import TimeDelta

        remaining = cutoff
        nleft     = self.max_invocations

        while (remaining is None or remaining > TimeDelta()) and nleft > 0:
            (action, pair)     = self._solve_once_on(task, remaining, random)
            (outcome, result)  = pair
            nleft             -= 1

            if remaining is not None:
                remaining -= action.cost

            if result.satisfiable is not None:
                return (result.satisfiable, result.certificate)

        return (None, None)

    def _solve_once_on(self, task, remaining, random):
        """
        Evaluate once on a specific task.
        """

        # select an action
        action_generator = self.strategy.select(task, remaining)
        action           = action_generator.send(None)

        if action is None:
            return (None, None)

        # take it, and provide the outcome
        (outcome, result) = action.take(task, random)

        try:
            action_generator.send(outcome)
        except StopIteration:
            pass

        return (action, (outcome, result))

