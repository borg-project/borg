"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools                      import assert_equal
from utexas.sat.solvers.test.support import FixedSolver

def test_named_solver():
    """
    Test the lookup-based SAT solver.
    """

    # set up the solver
    from utexas.sat.solvers import (
        SAT_Environment,
        SAT_LookupSolver,
        )

    certificate   = [1, 2, 3, 4, 0]
    foo_solver    = FixedSolver(True, certificate)
    named_solvers = {
        "foo": foo_solver,
        }
    environment   = SAT_Environment(named_solvers = named_solvers)
    solver        = SAT_LookupSolver("foo")

    # test it
    result = solver.solve(None, None, None, environment)

    assert_equal(result.satisfiable, foo_solver.satisfiable)
    assert_equal(result.certificate, foo_solver.certificate)

