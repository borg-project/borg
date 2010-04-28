"""
utexas/sat/solvers/mock_competition.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log               import get_logger
from utexas.sat.solvers.base import SAT_Solver

log = get_logger(__name__)

#         sat_case  = [(SAT_SolverRun.proc_elapsed <= action.cutoff, SAT_SolverRun.satisfiable)]
#         statement = \
#             select(
#                 [
#                     case(sat_case),
#                     SAT_SolverRun.proc_elapsed,
#                     ],
#                 and_(
#                     SAT_SolverRun.task_uuid   == task.task.uuid,
#                     SAT_SolverRun.solver_name == action.solver.name,
#                     SAT_SolverRun.cutoff      >= action.cutoff,
#                     ),
#                 order_by = sql_random(),
#                 limit    = 1,
#                 )

#         with closing(self.LocalResearchSession()) as l_session:
#             ((sat, elapsed),) = l_session.execute(statement)

#             return (SAT_Outcome.BY_VALUE[sat], min(elapsed, action.cutoff))

class SAT_MockCompetitionSolver(SAT_Solver):
    """
    Fake competition solver behavior by recycling past data.
    """

    def __init__(self, solver_name, engine):
        """
        Initialize.
        """

        from sqlalchemy.orm import sessionmaker

        SAT_Solver.__init__(self)

        self.solver_name          = solver_name
        self.LocalResearchSession = sessionmaker(bind = engine)

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
            CPU_LimitedRunRow as CLR,
            SAT_RunAttemptRow as SRA,
            )

        from_ = task.select_attempts().alias()
        query = \
            select(
                SRA.__table__.columns,
                and_(
                    from_.c.uuid     == SRA.__table__.c.uuid,
                    SRA.solver_name  == self.solver_name,
                    SRA.run_uuid     == CLR.uuid,
                    CLR.cutoff       >= cutoff,
#                     CLR.proc_elapsed <= cutoff,
                    ),
                order_by = sql_random(),
                limit    = 1,
                )

        # execute the query
        from contextlib import closing

        with closing(self.LocalResearchSession()) as session:
            print list(session.execute(query))

