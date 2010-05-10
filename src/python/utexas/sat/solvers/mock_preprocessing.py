"""
utexas/sat/solvers/preprocessing.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log               import get_logger
from utexas.sat.solvers.base import SAT_Solver

log = get_logger(__name__)

class SAT_MockPreprocessingSolver(SAT_Solver):
    """
    Execute a solver after a preprocessor pass.
    """

    def __init__(self, preprocessor_name, solver, engine):
        """
        Initialize.
        """

        from sqlalchemy.orm import sessionmaker

        SAT_Solver.__init__(self)

        self.preprocessor_name    = preprocessor_name
        self.solver               = solver
        self.engine               = engine
        self.LocalResearchSession = sessionmaker(bind = self.engine)

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
        from contextlib               import closing
        from sqlalchemy.sql.functions import random as sql_random
        from utexas.sat.solvers.base  import SAT_BareResult

        pre_query = query.order_by(sql_random()).limit(1)

        with closing(self.LocalResearchSession()) as session:
            row = session.execute(pre_query)

            ((inner_attempt_uuid, preprocessed, satisfiable, certificate_blob, elapsed),) = row

            if elapsed > cutoff:
                return SAT_BareResult(None, None)
            elif inner_attempt_uuid is None:
                if certificate_blob is not None:
                    certificate = SA.unpack_certificate(certificate_blob)
                else:
                    certificate = None

                return SAT_BareResult(satisfiable, certificate)

        # not solved by the preprocessor; try the inner solver
        from utexas.sat.tasks import SAT_MockPreprocessedTask

        if preprocessed:
            outer_task = SAT_MockPreprocessedTask(self.preprocessor_name, task, query)
        else:
            outer_task = task

        return self.solver.solve(outer_task, cutoff, seed)

