"""
utexas/sat/solvers/sanitizing.py

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

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Attempt to solve the specified instance; return the outcome.
        """

        from os       import fsync
        from os.path  import join
        from cargo.io import mkdtemp_scoped

        log.info("starting to solve %s", input_path)

        # FIXME use a temporary file, not directory

        with mkdtemp_scoped(prefix = "sanitized.") as sandbox_path:
            # unconditionally sanitize the instance
            from utexas.sat.cnf import write_sanitized_cnf

            sanitized_path = join(sandbox_path, "sanitized.cnf")

            with open(input_path) as input_path:
                with open(sanitized_path, "w") as sanitized_file:
                    write_sanitized_cnf(input_path, sanitized_file)
                    sanitized_file.flush()
                    fsync(sanitized_file.fileno())

            log.info("sanitized task is %s", sanitized_path)

            # execute the next solver in the chain
            return self.solver.solve(sanitized_path, cutoff, seed)

