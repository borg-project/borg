"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from contextlib   import contextmanager
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

        with self._uncompressed_task(task) as inner_task:
            return self._inner.solve(inner_task, budget, random, environment)

    @contextmanager
    def _uncompressed_task(self, task):
        """
        Provide an uncompressed task in a managed context.
        """

        # argument sanity
        from borg.tasks import AbstractFileTask

        if not isinstance(task, AbstractFileTask):
            raise TypeError("uncompressing solver requires a file-backed task")

        # create the context
        from cargo.io import mkdtemp_scoped

        with mkdtemp_scoped(prefix = "uncompressing.") as sandbox_path:
            # decompress the instance, if necessary
            from os.path  import join
            from cargo.io import decompress_if

            sandboxed_path    = join(sandbox_path, "uncompressed.cnf")
            uncompressed_path = decompress_if(task.path, sandboxed_path)

            log.info("uncompressed task file is %s", uncompressed_path)

            # provide the task
            from borg.tasks import UncompressedFileTask

            yield UncompressedFileTask(uncompressed_path, task)

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        return self._inner.get_new_row(session)

class UncompressingPreprocessor(UncompressingSolver, AbstractPreprocessor):
    """
    Uncompress and then preprocess SAT instances.
    """

    def preprocess(self, task, budget, output_path, random, environment):
        """
        Preprocess an instance.
        """

        with self._uncompressed_task(task) as inner_task:
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

