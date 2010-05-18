"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools                      import assert_equal
from utexas.sat.solvers.test.support import FixedSolver

def test_sat_portfolio_solver():
    """
    Test the SAT solver portfolio shell.
    """

    # set up the portfolio solver
    from cargo.temporal              import TimeDelta
    from utexas.sat.solvers          import (
        SAT_Environment,
        SAT_PortfolioSolver,
        )
    from utexas.portfolio.sat_world  import SAT_WorldAction
    from utexas.portfolio.strategies import SequenceSelectionStrategy

    certificate = [1, 2, 3, 4, 0]
    subsolvers  = [
        FixedSolver(None,  None),
        FixedSolver(True,  certificate),
        FixedSolver(False, None),
        ]
    actions     = [SAT_WorldAction(s, TimeDelta(seconds = 16.0)) for s in subsolvers]
    environment = SAT_Environment()

    # each test is similar
    def test_now(seconds, satisfiable, certificate, clean_record):
        """
        Test the SAT solver portfolio shell.
        """

        import numpy

        from utexas.sat.tasks import FileTask

        strategy = SequenceSelectionStrategy(actions)
        solver   = SAT_PortfolioSolver(strategy)
        task     = FileTask("/tmp/arbitrary_path.cnf")
        result   = solver.solve(task, TimeDelta(seconds = seconds), numpy.random, environment)

        assert_equal(result.satisfiable, satisfiable)
        assert_equal(result.certificate, certificate)
        assert_equal([s for (s, _) in result.record], clean_record)

    # yield the individual tests
    yield (test_now,  2.0, None, None,        [])
    yield (test_now, 18.0, None, None,        subsolvers[:1])
    yield (test_now, 34.0, True, certificate, subsolvers[:2])
    yield (test_now, 72.0, True, certificate, subsolvers[:2])

