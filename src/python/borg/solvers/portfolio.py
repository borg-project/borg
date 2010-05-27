"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from borg.rowed   import Rowed
from borg.solvers import (
    RunAttempt,
    AbstractSolver,
    )

# FIXME hack---shouldn't really be a run attempt

class PortfolioAttempt(RunAttempt):
    """
    Result of a portfolio solver.
    """

    def __init__(self, solver, task, budget, cost, record):
        """
        Initialize.
        """

        from cargo.unix.accounting import CPU_LimitedRun

        if record:
            (_, attempt) = record[-1]
            answer       = attempt.answer
        else:
            answer = None

        RunAttempt.__init__(
            self,
            solver,
            task,
            answer,
            None,
            CPU_LimitedRun(None, budget, None, None, None, cost, None, None),
            )

        self._record = record

    @property
    def record(self):
        """
        List of (subsolver, result) pairs.
        """

        return self._record

class PortfolioSolver(Rowed, AbstractSolver):
    """
    Solve tasks with a portfolio.
    """

    def __init__(self, strategy):
        """
        Initialize.
        """

        Rowed.__init__(self)

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
        record    = []

        while remaining > TimeDelta() and nleft > 0:
            (action, result)   = self._solve_once_on(task, remaining, random, environment)
            nleft             -= 1

            if action is None:
                break
            else:
                remaining = TimeDelta.from_timedelta(remaining - action.budget)

                record.append((action.solver, result))

                if result.answer is not None:
                    break

        return PortfolioAttempt(self, task, budget, budget - remaining, record)

    def _solve_once_on(self, task, budget, random, environment):
        """
        Evaluate once on a specific task.
        """

        # select an action
        action_generator = self.strategy.select(task, budget.as_s, random)
        action           = action_generator.send(None)

        if action is None:
            return (None, None)
        if action.budget > budget:
            raise RuntimeError("strategy selected an infeasible action")

        # take it, and provide the outcome
        from cargo.temporal                import TimeDelta
        from borg.portfolio.decision_world import DecisionWorldOutcome

        calibrated = TimeDelta(seconds = action.cost * environment.time_ratio)
        result     = action.solver.solve(task, calibrated, random, environment)
        outcome    = DecisionWorldOutcome.from_result(result)

        try:
            action_generator.send(outcome)
        except StopIteration:
            pass

        return (action, result)

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        from borg.data import SolverRow

        solver_row = session.query(SolverRow).get(self.name)

        if solver_row is None:
            solver_row = SolverRow(name = self.name, type = "sat")

            session.add(solver_row)

        return solver_row

    @property
    def name(self):
        """
        A name for this solver.
        """

        return "portfolio"

    @staticmethod
    def build(request, trainer):
        """
        Build a solver as requested.
        """

        from borg.portfolio.strategies import build_strategy

        return PortfolioSolver(build_strategy(request, trainer))

