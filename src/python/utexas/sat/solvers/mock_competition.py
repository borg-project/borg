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

        # build the query
        from sqlalchemy              import (
            and_,
            select,
            )
        from sqlalchemy.sql.functions import random as sql_random
        from utexas.data              import (
            SAT_AttemptRow    as SA,
            SAT_RunAttemptRow as SRA,
            )

        from_ = task.select_attempts().alias()
        query = \
            select(
                [
                    from_.c.cost,
                    from_.c.satisfiable,
                    from_.c.certificate,
                    SRA.seed,
                    ],
                and_(
                    from_.c.budget   >= budget,
                    from_.c.uuid     == SRA.__table__.c.uuid,
                    SRA.solver_name  == self.solver_name,
                    ),
                order_by = sql_random(),
                limit    = 1,
                )

        # execute the query
        with environment.CacheSession() as session:
            # unpack the result row, if any
            row = session.execute(query).first()

            if row is None:
                raise RuntimeError("no matching attempt row in database")
            else:
                (cost, satisfiable, certificate_blob, seed) = row

            # interpret the result row
            if cost <= budget:
                if certificate_blob is None:
                    certificate = None
                elif satisfiable:
                    certificate = SA.unpack_certificate(certificate_blob)
                else:
                    raise RuntimeError("non-sat row has non-null certificate")

                return SAT_MockCompetitionResult(self, task, budget, cost, satisfiable, certificate, seed)
            else:
                return SAT_MockCompetitionResult(self, task, budget, cost, None, None, seed)

    def to_orm(self, session):
        """
        Return a database description of this solver.
        """

        from utexas.data import SAT_SolverRow

        return session.query(SAT_SolverRow).filter_by(name = self.solver_name).first()

