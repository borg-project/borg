"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from borg.rowed   import Rowed
from borg.solvers import (
    AbstractSolver,
    AbstractPreprocessor,
    )

class LookupSolver(Rowed, AbstractSolver):
    """
    A solver which indirectly executes a named solver.
    """

    def __init__(self, name):
        """
        Initialize.
        """

        Rowed.__init__(self)

        self._name = name

    def solve(self, task, budget, random, environment):
        """
        Look up the named solver and use it on the task.
        """

        from borg.solvers.attempts import (
            WrappedAttempt,
            WrappedRunAttempt,
            AbstractRunAttempt,
            )

        looked_up = self.look_up(environment)
        inner     = looked_up.solve(task, budget, random, environment)

        if isinstance(inner, AbstractRunAttempt):
            return WrappedRunAttempt(self, inner)
        else:
            return WrappedAttempt(self, inner)

    def look_up(self, environment):
        """
        Look up the named solvers.
        """

        return environment.named_solvers[self._name]

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        from borg.data import SolverRow

        solver_row = session.query(SolverRow).get(self._name)

        if solver_row is None:
            solver_row = SolverRow(name = self._name, type = "sat")

            session.add(solver_row)

        return solver_row

    def get_seeded(self, environment):
        """
        Is the solver seeded?
        """

        return self.look_up(environment).seeded

    @property
    def name(self):
        """
        The name used for lookup by this solver.
        """

        return self._name

    @staticmethod
    def build(trainer, request):
        """
        Build a solver as requested.
        """

        return LookupSolver(request["name"])

class LookupPreprocessor(LookupSolver, AbstractPreprocessor):
    """
    A preprocessor which indirectly executes a named preprocessor.
    """

    def preprocess(self, task, budget, output_path, random, environment):
        """
        Preprocess an instance.
        """

        from borg.solvers.attempts import WrappedPreprocessorAttempt

        looked_up = self.look_up(environment)
        inner     = looked_up.preprocess(task, budget, output_path, random, environment)

        return WrappedPreprocessorAttempt(self, inner)

    def extend(self, task, answer, environment):
        """
        Extend an answer.
        """

        looked_up = self.look_up(environment)

        return looked_up.extend(task, answer, environment)

    def make_task(self, seed, input_task, output_path, environment, row = None):
        """
        Construct an appropriate preprocessed task from its output directory.
        """

        from borg.tasks import (
            WrappedPreprocessedTask,
            AbstractPreprocessedTask,
            WrappedPreprocessedDirectoryTask,
            AbstractPreprocessedDirectoryTask,
            )

        looked_up = self.look_up(environment)
        task      = looked_up.make_task(seed, input_task, output_path, environment, row = row)

        if isinstance(task, AbstractPreprocessedDirectoryTask):
            return WrappedPreprocessedDirectoryTask(self, task)
        else:
            return WrappedPreprocessedTask(self, task)

