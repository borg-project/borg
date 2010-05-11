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

    def __init__(self, solver, task, budget, cost, satisfiable, certificate):
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

    def to_orm(self):
        """
        Return a database description of this result.
        """

        attempt_row = \
            SAT_RunAttemptRow(
                run    = \
                    CPU_LimitedRunRow(
                        cutoff = self.budget,
                        proc_elapsed = self.cost,
                        ),
                solver = self.solver.to_orm(),
                seed   = self.seed,
                )

        return self.update_orm(attempt_row)

class SAT_MockCompetitionSolver(SAT_Solver):
    """
    Fake competition solver behavior by recycling past data.
    """

    def __init__(self, solver_name, engine):
        """
        Initialize.
        """

        from cargo.sql.alchemy import session_maker

        SAT_Solver.__init__(self)

        self.solver_name = solver_name
        self.Session     = session_maker(bind = engine)

    def solve(self, task, cutoff = None, seed = None):
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
                    ],
                and_(
                    from_.c.budget   >= cutoff,
                    from_.c.uuid     == SRA.__table__.c.uuid,
                    SRA.solver_name  == self.solver_name,
                    ),
                order_by = sql_random(),
                limit    = 1,
                )

        # execute the query
        with self.Session() as session:
            ((cost, satisfiable, certificate_blob),) = session.execute(query)

            if cost <= cutoff:
                if satisfiable:
                    certificate = SA.unpack_certificate(certificate_blob)
                elif certificate_blob is not None:
                    raise RuntimeError("non-sat row has non-null certificate")
                else:
                    certificate = None

                return SAT_MockCompetitionResult(self, task, cutoff, cost, satisfiable, certificate)
            else:
                return SAT_MockCompetitionResult(self, task, cutoff, cost, None, None)

    def to_orm(self):
        """
        Return a database description of this solver.
        """

        return SAT_SolverRow(name = self.solver_name)

