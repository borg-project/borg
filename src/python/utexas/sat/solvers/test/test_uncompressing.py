"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_uncompressing_solver():
    """
    Test the uncompressing solver wrapper.
    """

    from tempfile                        import NamedTemporaryFile
    from utexas.sat.solvers.test.support import (
        TaskVerifyingSolver,
        unsanitized_cnf,
        )

    with NamedTemporaryFile(suffix = ".cnf.gz") as named_file:
        # write the compressed CNF expression
        from gzip       import GzipFile
        from contextlib import closing

        with closing(GzipFile(mode = "w", fileobj = named_file)) as gzip_file:
            gzip_file.write(unsanitized_cnf)

        named_file.flush()

        # test the solver
        from utexas.sat.tasks   import SAT_FileTask
        from utexas.sat.solvers import SAT_UncompressingSolver

        inner_solver = TaskVerifyingSolver(unsanitized_cnf)
        solver       = SAT_UncompressingSolver(inner_solver)
        task         = SAT_FileTask(named_file.name)

        solver.solve(task, None, None, None)

