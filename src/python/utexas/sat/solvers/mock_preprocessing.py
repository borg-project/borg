"""
utexas/sat/solvers/preprocessing.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log               import get_logger
from utexas.sat.solvers.base import (
    SAT_Solver,
    SAT_BareResult,
    )

log = get_logger(__name__)

class SAT_MockPreprocessingResult(SAT_BareResult):
    """
    Outcome of a simulated preprocessing solver.
    """

    def __init__(self, solver, task, budget, cost, satisfiable, certificate, inner_result):
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

        self.inner_result = inner_result

    def to_orm(self):
        """
        Return a database description of this result.
        """

        if self.inner_result is None:
            inner_attempt_row = None
        else:
            inner_attempt_row = self.inner_result.to_orm()

        preprocessor_row = SAT_PreprocessorRow(name = self.solver.preprocessor_name)
        attempt_row      = \
            SAT_PreprocessingAttemptRow(
                run          = \
                    CPU_LimitedRunRow(
                        cutoff       = self.budget,
                        proc_elapsed = self.cost,
                        ),
                preprocessor      = preprocessor_row,
                inner_attempt_row = inner_attempt_row,
                )

        return self.update_orm(attempt_row)

class SAT_MockPreprocessingSolver(SAT_Solver):
    """
    Execute a solver after a preprocessor pass.
    """

    def __init__(self, preprocessor_name, solver, engine):
        """
        Initialize.
        """

        from cargo.sql.alchemy import session_maker

        SAT_Solver.__init__(self)

        self.preprocessor_name = preprocessor_name
        self.solver            = solver
        self.engine            = engine
        self.Session           = session_maker(bind = self.engine)

    def solve(self, task, cutoff = None, seed = None):
        """
        Execute the solver and return its outcome, given an input task.
        """

        # argument sanity
        if cutoff is None:
            raise ValueError("cutoff required")

        # build the preprocessor-run query
        from sqlalchemy  import (
            and_,
            select,
            )
        from utexas.data import (
            SAT_AttemptRow              as SA,
            CPU_LimitedRunRow           as CLR,
            SAT_PreprocessingAttemptRow as SPA,
            )

        from_ = task.select_attempts().alias()
        query = \
            select(
                [
                    SPA.inner_attempt_uuid,
                    SPA.preprocessed,
                    from_.c.satisfiable,
                    from_.c.certificate,
                    CLR.proc_elapsed,
                    ],
                and_(
                    from_.c.budget        >= cutoff,
                    from_.c.uuid          == SPA.__table__.c.uuid,
                    SPA.preprocessor_name == self.preprocessor_name,
                    SPA.run_uuid          == CLR.uuid,
                    ),
                )

        # select a preprocessor run
        from sqlalchemy.sql.functions import random as sql_random

        pre_query = query.order_by(sql_random()).limit(1)

        with self.Session() as session:
            row = session.execute(pre_query)

            ((inner_attempt_uuid, preprocessed, satisfiable, certificate_blob, elapsed),) = row

            if elapsed > cutoff:
                return SAT_MockPreprocessingResult(self, task, cutoff, elapsed, None, None, None)
            elif inner_attempt_uuid is None:
                if certificate_blob is not None:
                    certificate = SA.unpack_certificate(certificate_blob)
                else:
                    certificate = None

                return SAT_MockPreprocessingResult(self, task, cutoff, elapsed, satisfiable, certificate, None)

        # not solved by the preprocessor; try the inner solver
        from utexas.sat.tasks import SAT_MockPreprocessedTask

        if preprocessed:
            outer_task = SAT_MockPreprocessedTask(self.preprocessor_name, task, query)
        else:
            outer_task = task

        inner_result = self.solver.solve(outer_task, cutoff - elapsed, seed)

        return \
            SAT_MockPreprocessingResult(
                self,
                task,
                cutoff,
                inner_result.cost + elapsed,
                inner_result.satisfiable,
                inner_result.certificate,
                inner_result,
                )

