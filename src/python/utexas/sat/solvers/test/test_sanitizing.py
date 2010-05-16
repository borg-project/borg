"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_sat_sanitizing_solver():
    """
    Test the sanitizing SAT solver wrapper.
    """

    from tempfile                        import NamedTemporaryFile
    from utexas.sat.solvers.test.support import (
        TaskVerifyingSolver,
        sanitized_cnf,
        unsanitized_cnf,
        )

    with NamedTemporaryFile(suffix = ".cnf") as named_file:
        # write the unsanitized CNF
        named_file.write(raw_unsanitized_cnf)
        named_file.flush()

        # test the solver
        from utexas.sat.tasks   import SAT_FileTask
        from utexas.sat.solvers import SAT_SanitizingSolver

        inner_solver = TaskVerifyingSolver(sanitized_cnf)
        solver       = SAT_SanitizingSolver(inner_solver)
        task         = SAT_FileTask(named_file.name)

        solver.solve(task, None, None, None)

