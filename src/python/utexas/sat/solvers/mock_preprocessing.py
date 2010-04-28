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

        # build the query
        from sqlalchemy       import (
            and_,
            select,
            )
        from utexas.data      import (
            CPU_LimitedRunRow           as CLR,
            SAT_PreprocessingAttemptRow as SPA,
            )

        from_ = task.select_attempts().alias()
        query = \
            select(
                SPA.__table__.columns,
                and_(
                    from_.c.uuid          == SPA.__table__.c.uuid,
                    SPA.preprocessor_name == self.preprocessor_name,
                    SPA.run_uuid          == CLR.uuid,
                    CLR.cutoff            >= cutoff,
#                     CLR.proc_elapsed      <= cutoff,
                    ),
                )

        # grab a matching run
        from contextlib import closing

        # FIXME need to actually select a single row, etc.

        from utexas.sat.tasks import SAT_MockPreprocessedTask

        outer_task = SAT_MockPreprocessedTask(self.preprocessor_name, task, query)

        return self.solver.solve(outer_task, cutoff, seed)

#         with closing(self.LocalResearchSession()) as session:
#             rows = session.execute(query)

#             for row in rows:
#                 print row

