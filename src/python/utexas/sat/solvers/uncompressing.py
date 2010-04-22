"""
utexas/sat/solvers/uncompressing.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.solvers.base import SAT_Solver

class SAT_UncompressingSolver(SAT_Solver):
    """
    Execute another solver using an uncompressed instance.
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

        from os.path  import join
        from cargo.io import (
            decompress_if,
            mkdtemp_scoped,
            )

        # FIXME only create the temporary directory if necessary

        with mkdtemp_scoped(prefix = "solver_input.") as sandbox_path:
            # decompress the instance, if necessary
            uncompressed_path = \
                decompress_if(
                    input_path,
                    join(sandbox_path, "uncompressed.cnf"),
                    )

            log.info("uncompressed task is %s", uncompressed_path)

            # execute the next solver in the chain
            return self.solver.solve(uncompressed_path, cutoff, seed)

