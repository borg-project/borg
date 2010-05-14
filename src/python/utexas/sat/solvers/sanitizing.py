"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log               import get_logger
from utexas.sat.solvers.base import SAT_Solver

log = get_logger(__name__)

class SAT_SanitizingSolver(SAT_Solver):
    """
    Execute another solver using a sanitized instance.
    """

    def __init__(self, solver):
        """
        Initialize.
        """

        SAT_Solver.__init__(self)

        self.solver = solver

    def solve(self, task, budget, random, environment):
        """
        Attempt to solve the specified instance; return the outcome.
        """

        from os.path  import join
        from cargo.io import mkdtemp_scoped

        # argument sanity
        from utexas.sat.tasks import SAT_FileTask

        if not isinstance(task, SAT_FileTask):
            raise TypeError("sanitizing solver requires a file-backed task")

        with mkdtemp_scoped(prefix = "sanitized.") as sandbox_path:
            # unconditionally sanitize the instance
            from utexas.sat.cnf import write_sanitized_cnf

            sanitized_path = join(sandbox_path, "sanitized.cnf")

            with open(task.path) as task_file:
                with open(sanitized_path, "w") as sanitized_file:
                    write_sanitized_cnf(task_file, sanitized_file)

                    sanitized_file.flush()

            log.info("sanitized task file is %s", sanitized_path)

            # execute the next solver in the chain
            sanitized_task = SAT_FileTask(sanitized_path)

            return self.solver.solve(sanitized_task, budget, random, environment)

