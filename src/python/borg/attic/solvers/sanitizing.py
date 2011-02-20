"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log    import get_logger
from borg.rowed   import Rowed
from borg.solvers import AbstractSolver

log = get_logger(__name__)

class SanitizingSolver(Rowed, AbstractSolver):
    """
    Execute another solver using a sanitized instance.
    """

    def __init__(self, inner):
        """
        Initialize.
        """

        Rowed.__init__(self)

        self._inner = inner

    def solve(self, task, budget, random, environment):
        """
        Attempt to solve the specified instance; return the outcome.
        """

        from os.path  import join
        from cargo.io import mkdtemp_scoped

        # argument sanity
        from borg.tasks import AbstractFileTask

        if not isinstance(task, AbstractFileTask):
            raise TypeError("sanitizing solver requires a file-backed task")

        with mkdtemp_scoped(prefix = "sanitized.") as sandbox_path:
            # unconditionally sanitize the instance
            from borg.sat.cnf import write_sanitized_cnf

            sanitized_path = join(sandbox_path, "sanitized.cnf")

            with open(task.path) as task_file:
                with open(sanitized_path, "w") as sanitized_file:
                    write_sanitized_cnf(task_file, sanitized_file)

                    sanitized_file.flush()

            log.info("sanitized task file is %s", sanitized_path)

            # execute the next solver in the chain
            from borg.tasks import FileTask

            sanitized_task = FileTask(sanitized_path)

            return self._inner.solve(sanitized_task, budget, random, environment)

