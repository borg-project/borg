"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log    import get_logger
from borg.rowed   import Rowed
from borg.solvers import (
    AbstractSolver,
    AbstractPreprocessor,
    )

log = get_logger(__name__)

class UncompressingSolver(Rowed, AbstractSolver):
    """
    Execute another solver using an uncompressed instance.
    """

    def __init__(self, solver):
        """
        Initialize.
        """

        Rowed.__init__(self)

        self._inner = solver

    def solve(self, task, budget, random, environment):
        """
        Attempt to solve the specified instance; return the outcome.
        """

        from borg.tasks import uncompressed_task

        with uncompressed_task(task) as inner_task:
            return self._inner.solve(inner_task, budget, random, environment)

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        return self._inner.get_new_row(session)

    @property
    def name(self):
        """
        Get the name of this solver, if any.
        """

        return self._inner.name

class UncompressingPreprocessor(UncompressingSolver, AbstractPreprocessor):
    """
    Uncompress and then preprocess SAT instances.
    """

    def preprocess(self, task, budget, output_path, random, environment):
        """
        Preprocess an instance.
        """

        from borg.tasks import uncompressed_task

        with uncompressed_task(task) as inner_task:
            return self._inner.preprocess(inner_task, budget, output_path, random, environment)

    def extend(self, task, answer):
        """
        Pretend to extend an answer.
        """

        return self._inner.extend(task, answer)

    def make_task(self, seed, input_task, output_path, environment, row = None):
        """
        Construct an appropriate preprocessed task from its output directory.
        """

        return self._inner.make_task(seed, input_task, output_path, environment, row = row)

