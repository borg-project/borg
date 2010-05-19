"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_sat_sanitizing_solver():
    """
    Test the sanitizing SAT solver wrapper.
    """

    from tempfile                  import NamedTemporaryFile
    from borg.solvers.test.support import (
        TaskVerifyingSolver,
        sanitized_cnf,
        unsanitized_cnf,
        )

    with NamedTemporaryFile(suffix = ".cnf") as named_file:
        # write the unsanitized CNF
        named_file.write(unsanitized_cnf)
        named_file.flush()

        # test the solver
        from borg.tasks   import FileTask
        from borg.solvers import SanitizingSolver

        inner_solver = TaskVerifyingSolver(sanitized_cnf)
        solver       = SanitizingSolver(inner_solver)
        task         = FileTask(named_file.name)

        solver.solve(task, None, None, None)

