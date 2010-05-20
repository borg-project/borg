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
        from borg.tasks import AbstractTask

        assert isinstance(task, AbstractTask)

        # generate a recycled result
        with environment.CacheSession() as session:
            from borg.data    import RunAttemptRow
            from borg.solvers import Attempt

            attempt_row = self._get_attempt_row(session, task, budget, RunAttemptRow)

            if attempt_row.cost <= budget:
                return Attempt(self, budget, attempt_row.cost, task, attempt_row.get_answer())
            else:
                return Attempt(self, budget, budget, task, None)

    def _get_attempt_row(self, session, task, budget, AR):
        """
        Query the database for a matching attempt to recycle.
        """

        # mise en place
        from sqlalchemy               import and_
        from sqlalchemy.sql.functions import random as sql_random
        from borg.data                import TrialRow as TR

        # select an appropriate attempt to recycle
        task_row    = task.get_row(session)
        solver_row  = self.get_row(session)
        attempt_row =                                               \
            session                                                 \
            .query(AR)                                              \
            .filter(
                and_(
                    AR.task   == task_row,
                    AR.budget >= budget,
                    AR.solver == solver_row,
                    AR.trials.contains(TR.get_recyclable(session)),
                    )
                )                                                   \
            .order_by(sql_random())                                 \
            .first()

        if attempt_row is None:
            raise RuntimeError("database does not contain a matching recyclable run")
        else:
            return attempt_row

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        from borg.data import SolverRow

        solver_row = session.query(SolverRow).get(self._solver_name)

        if solver_row is None:
            solver_row = SolverRow(name = self._solver_name, type = "sat")

            session.add(solver_row)

        return solver_row

class RecyclingPreprocessor(RecyclingSolver):
    """
    Execute a solver after a preprocessor pass.
    """

    def __init__(self, preprocessor_name):
        """
        Initialize.
        """

        RecyclingSolver.__init__(self, preprocessor_name)

    def preprocess(self, task, budget, output_path, random, environment):
        """
        Preprocess the instance.
        """

        # argument sanity
        from borg.tasks import AbstractTask

        assert isinstance(task, AbstractTask)

        # generate a recycled result
        with environment.CacheSession() as session:
            from cargo.unix.accounting import CPU_LimitedRun
            from borg.data             import PreprocessorAttemptRow
            from borg.solvers          import PreprocessorAttempt

            attempt_row = self._get_attempt_row(session, task, budget, PreprocessorAttemptRow)

            if attempt_row.cost <= budget:
                output_task_row = attempt_row.output_task

                if attempt_row.output_task == attempt_row.task:
                    output_task = task
                else:
                    from borg.tasks import PreprocessedTask

                    output_task = \
                        PreprocessedTask(
                            self,
                            attempt_row.seed,
                            task,
                            row = attempt_row.output_task,
                            )

                return \
                    PreprocessorAttempt(
                        self,
                        task,
                        attempt_row.get_answer(),
                        attempt_row.seed,
                        CPU_LimitedRun(None, budget, None, None, None, attempt_row.cost, None, None),
                        output_task,
                        )
            else:
                return \
                    PreprocessorAttempt(
                        self,
                        task,
                        None,
                        attempt_row.seed,
                        CPU_LimitedRun(None, budget, None, None, None, budget, None, None),
                        task,
                        )

    def extend(self, task, answer, environment):
        """
        Extend an answer to a preprocessed task to its parent task.
        """

        return answer

    def make_task(self, seed, input_task, output_path, environment, row = None):
        """
        Construct an appropriate preprocessed task from its output directory.
        """

        raise NotImplementedError()

