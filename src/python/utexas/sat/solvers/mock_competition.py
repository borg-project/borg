"""
utexas/sat/solvers/mock_competition.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log               import get_logger
from utexas.sat.solvers.base import (
    SAT_Solver,
    SAT_BareResult,
    )

log = get_logger(__name__)

class SAT_MockCompetitionResult(SAT_BareResult):
    """
    Outcome of a simulated external SAT solver binary.
    """

    def __init__(self, solver, task, budget, cost, satisfiable, certificate, seed):
        """
        Initialize.
        """

        SAT_BareResult.__init__(
            self,
            solver,
            task,
            budget,
            cost,
            satisfiable,
            certificate,
            )

        self.seed = seed

    def to_orm(self, session):
        """
        Return a database description of this result.
        """

        from utexas.data import (
            SAT_RunAttemptRow,
            CPU_LimitedRunRow,
            )

        attempt_row = \
            self.update_orm(
                session,
                SAT_RunAttemptRow(
                    run    = \
                        CPU_LimitedRunRow(
                            cutoff       = self.budget,
                            proc_elapsed = self.cost,
                            ),
                    solver = self.solver.to_orm(session),
                    seed   = self.seed,
                    ),
                )

        session.add(attempt_row)

        return attempt_row

class SAT_MockCompetitionSolver(SAT_Solver):
    """
    Fake competition solver behavior by recycling past data.
    """

    def __init__(self, solver_name):
        """
        Initialize.
        """

        SAT_Solver.__init__(self)

        self.solver_name = solver_name

    def solve(self, task, budget, random, environment):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        # argument sanity
        from utexas.sat.tasks import MockTask

        if not isinstance(task, MockTask):
            raise TypeError("mock solvers require mock tasks")

        # mise en place
        from sqlalchemy               import (
            and_,
            )
        from sqlalchemy.sql.functions import random as sql_random
        from utexas.data              import (
            SAT_TrialRow      as ST,
            SAT_AttemptRow    as SA,
            SAT_RunAttemptRow as SRA,
            )

        # generate a recycled result
        with environment.CacheSession() as session:
            # select an appropriate attempt to recycle
            attempt_row          =                           \
                session                                      \
                .query(SRA)                                  \
                .filter(
                    and_(
                        SRA.task_uuid        == task.task_uuid,
                        SRA.budget           >= budget,
                        SRA.solver_name      == self.solver_name,
                        SRA.trials.contains(ST.get_recyclable(session)),
                        )
                    )                                        \
                .order_by(sql_random())                      \
                .first()

            if attempt_row is None:
                raise RuntimeError("database does not contain a matching recyclable run")

            # interpret the attempt
            if attempt_row.cost <= budget:
                if attempt_row.answer is None:
                    satisfiable = None
                    certificate = None
                else:
                    satisfiable = attempt_row.answer.satisfiable
                    certificate = attempt_row.answer.get_certificate()

                return \
                    SAT_MockCompetitionResult(
                        self,
                        task,
                        budget,
                        attempt_row.cost,
                        satisfiable,
                        certificate,
                        attempt_row.seed,
                        )
            else:
                return SAT_MockCompetitionResult(self, task, budget, budget, None, None, attempt_row.seed)

    def to_orm(self, session):
        """
        Return a database description of this solver.
        """

        from utexas.data import SAT_SolverRow

        return session.query(SAT_SolverRow).get(self.solver_name)

