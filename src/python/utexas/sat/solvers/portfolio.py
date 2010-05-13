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

    def __init__(self, solver, task, budget, cost, results):
        """
        Initialize.
        """

        if results:
            last_result = results[-1]
            satisfiable = last_result.satisfiable
            certificate = last_result.certificate

        SAT_BareResult.__init__(
            self,
            solver,
            )

        self._results = results

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

    def solve(self, task, budget, random, environment):
        """
        Execute the solver and return its outcome, given an input path.
        """

        from cargo.temporal import TimeDelta

        # solve the instance
        remaining = budget
        nleft     = self.max_invocations

        while remaining > TimeDelta() and nleft > 0:
            (action, result)   = self._solve_once_on(task, remaining, random, environment)
            nleft             -= 1

            if action is None:
                break
            else:
                remaining -= action.cost

                if result.satisfiable is not None:
                    return result

        return SAT_BareResult(self, task, budget, budget - remaining, None, None)

    def _solve_once_on(self, task, budget, random, environment):
        """
        Evaluate once on a specific task.
        """

        # select an action
        action_generator = self.strategy.select(task, budget)
        action           = action_generator.send(None)

        if action is None:
            return (None, None)
        if action.cost > budget:
            raise RuntimeError("strategy selected an infeasible action")

        # take it, and provide the outcome
        from utexas.portfolio.sat_world import SAT_WorldOutcome

        result  = action.solver.solve(task, budget, random, environment)
        outcome = SAT_WorldOutcome.from_result(result)

        try:
            action_generator.send(outcome)
        except StopIteration:
            pass

        return (action, result)

