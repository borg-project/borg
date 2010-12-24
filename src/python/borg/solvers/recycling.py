"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import borg
import base

class RecycledAttempt(object):
    """
    The outcome of a solver's attempt on a task.
    """

    def __init__(self, solver, budget, cost, task, answer):
        """
        Initialize.
        """

        self.solver = solver
        self.budget = budget
        self.cost   = cost
        self.task   = task
        self.answer = answer

class RecyclingSolver(borg.Rowed, base.AbstractSolver):
    """
    Fake competition solver behavior by recycling past data.
    """

    def __init__(self, solver_name):
        """
        Initialize.
        """

        borg.Rowed.__init__(self)

        self._solver_name = solver_name

    def solve(self, task, budget, random, environment):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        from cargo.random import grab

        attempts = environment.attempts_cache[self._solver_name, task.uuid]
        compatible = [a for a in attempts if a.budget >= budget]
        grabbed = grab(compatible, random)
        answer = grabbed.answer_uuid if grabbed.cost <= budget else None

        return RecycledAttempt(self, task, budget, grabbed.cost, answer)

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

class RecyclingPreprocessor(RecyclingSolver, base.AbstractPreprocessor):
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
                        attempt_row.get_short_answer(),
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

        from borg.tasks import PreprocessedTask

        return PreprocessedTask(self, seed, input_task, row = row)

