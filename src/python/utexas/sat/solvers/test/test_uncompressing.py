"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools         import assert_equal
from utexas.sat.solvers import (
    SAT_Solver,
    SAT_BareResult,
    )

raw_cnf = \
"""c comment
c foo
p cnf 2 6
 -1 2 3 0
-4 -5 6 0
%
0
"""

class TaskVerifyingSolver(SAT_Solver):
    """
    Solver that merely verifies the contents of the task.
    """

    def __init__(self, correct_cnf):
        """
        Initialize.
        """

        self.correct_cnf = correct_cnf

    def solve(self, task, budget, random, environment):
        """
        Verify behavior.
        """

        with open(task.path) as task_file:
            assert_equal(task_file.read(), self.correct_cnf)

        return SAT_BareResult(self, task, budget, budget, None, None)

def test_sat_uncompressing_solver():
    """
    Test the uncompressing SAT solver wrapper.
    """

    # test
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix = ".cnf.gz") as named_file:
        # write the compressed CNF expression
        from gzip       import GzipFile
        from contextlib import closing

        with closing(GzipFile(mode = "w", fileobj = named_file)) as gzip_file:
            gzip_file.write(raw_cnf)

        named_file.flush()

        # test the solver
        from utexas.sat.tasks   import SAT_FileTask
        from utexas.sat.solvers import SAT_UncompressingSolver

        inner_solver = TaskVerifyingSolver(raw_cnf)
        solver       = SAT_UncompressingSolver(inner_solver)
        task         = SAT_FileTask(named_file.name)

        solver.solve(task, None, None, None)

