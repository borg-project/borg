"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log    import get_logger
from borg.rowed   import Rowed
from borg.solvers import (
    Attempt,
    AbstractSolver,
    )

log = get_logger(__name__)

# class SAT_MockCompetitionResult(SAT_BareResult):
#     """
#     Outcome of a simulated external SAT solver binary.
#     """

#     def __init__(self, solver, task, budget, cost, satisfiable, certificate, seed):
#         """
#         Initialize.
#         """

#         SAT_BareResult.__init__(
#             self,
#             solver,
#             task,
#             budget,
#             cost,
#             satisfiable,
#             certificate,
#             )

#         self.seed = seed

#     def to_orm(self, session):
#         """
#         Return a database description of this result.
#         """

#         from utexas.data import (
#             SAT_RunAttemptRow,
#             CPU_LimitedRunRow,
#             )

#         attempt_row = \
#             self.update_orm(
#                 session,
#                 SAT_RunAttemptRow(
#                     run    = \
#                         CPU_LimitedRunRow(
#                             cutoff       = self.budget,
#                             proc_elapsed = self.cost,
#                             ),
#                     solver = self.solver.to_orm(session),
#                     seed   = self.seed,
#                     ),
#                 )

#         session.add(attempt_row)

#         return attempt_row

class RecyclingSolver(Rowed, AbstractSolver):
    """
    Fake competition solver behavior by recycling past data.
    """

    def __init__(self, solver_name):
        """
        Initialize.
        """

        Rowed.__init__(self)

        self._solver_name = solver_name

    def solve(self, task, budget, random, environment):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        # argument sanity
#         from borg.tasks import RecycledTask

#         assert isinstance(task, RecycledTask)

        # mise en place
        from sqlalchemy               import and_
        from sqlalchemy.sql.functions import random as sql_random
        from borg.data                import (
            TrialRow      as TR,
            RunAttemptRow as RAR,
            )

        # generate a recycled result
        with environment.CacheSession() as session:
            # select an appropriate attempt to recycle
            task_row    = task.get_row(session)
            solver_row  = self.get_row(session)
            attempt_row =                                                \
                session                                                  \
                .query(AR)                                               \
                .filter(
                    and_(
                        RAR.task   == task_row,
                        RAR.budget >= budget,
                        RAR.solver == solver_row,
                        RAR.trials.contains(TR.get_recyclable(session)),
                        )
                    )                                                    \
                .order_by(sql_random())                                  \
                .first()

            if attempt_row is None:
                raise RuntimeError("database does not contain a matching recyclable run")

            # interpret the attempt
            from borg.solvers import Attempt

            if attempt_row.cost <= budget:
                return Attempt(self, budget, attempt_row.cost, task, attempt_row.answer)
            else:
                return Attempt(self, budget, budget, task, None)

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        from borg.data import SolverRow

        return session.query(SolverRow).get(self.solver_name)

