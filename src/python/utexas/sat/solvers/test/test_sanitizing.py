"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_sat_sanitizing_solver():
    """
    Test the sanitizing SAT solver wrapper.
    """

    # fake data
    from utexas.sat.solvers.test.test_uncompressing import (
        TaskVerifyingSolver,
        raw_cnf,
        )

    sanitized_cnf = \
"""p cnf 2 6
-1 2 3 0
-4 -5 6 0
"""

    # test
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix = ".cnf.gz") as named_file:
        # write the unsanitized CNF
        named_file.write(sanitized_cnf)
        named_file.flush()

        # test the solver
        from utexas.sat.tasks   import SAT_FileTask
        from utexas.sat.solvers import SAT_SanitizingSolver

        inner_solver = TaskVerifyingSolver(sanitized_cnf)
        solver       = SAT_SanitizingSolver(inner_solver)
        task         = SAT_FileTask(named_file.name)

        solver.solve(task, None, None, None)

