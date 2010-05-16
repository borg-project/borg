"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log               import get_logger
from utexas.sat.solvers.base import SAT_Solver

log = get_logger(__name__)

class SAT_UncompressingSolver(SAT_Solver):
    """
    Execute another solver using an uncompressed instance.
    """

    def __init__(self, solver, name = None):
        """
        Initialize.
        """

        SAT_Solver.__init__(self)

        self.solver = solver
        self.name   = name

    def solve(self, task, budget, random, environment):
        """
        Attempt to solve the specified instance; return the outcome.
        """

        from os.path  import join
        from cargo.io import (
            decompress_if,
            mkdtemp_scoped,
            )

        # argument sanity
        from utexas.sat.tasks import AbstractFileTask

        if not isinstance(task, AbstractFileTask):
            raise TypeError("uncompressing solver requires a file-backed task")

        # solver body
        with mkdtemp_scoped(prefix = "solver_input.") as sandbox_path:
            # decompress the instance, if necessary
            sandboxed_path    = join(sandbox_path, "uncompressed.cnf")
            uncompressed_path = decompress_if(task.path, sandboxed_path)

            log.info("uncompressed task file is %s", uncompressed_path)

            # execute the next solver in the chain
            from utexas.sat.tasks import FileTask

            uncompressed_task = FileTask(uncompressed_path)

            return self.solver.solve(uncompressed_task, budget, random, environment)

