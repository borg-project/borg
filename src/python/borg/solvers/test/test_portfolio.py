"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools                import assert_equal
from borg.solvers.test.support import FixedSolver

def test_sat_portfolio_solver():
    """
    Test the SAT solver portfolio shell.
    """

    # set up the portfolio solver
    from cargo.temporal            import TimeDelta
    from borg.sat                  import SAT_Answer
    from borg.solvers              import (
        Environment,
        PortfolioSolver,
        )
    from borg.portfolio.sat_world  import SAT_WorldAction
    from borg.portfolio.strategies import SequenceSelectionStrategy

    certificate = [1, 2, 3, 4, 0]
    subsolvers  = [
        FixedSolver(None),
        FixedSolver(SAT_Answer(True,  certificate)),
        FixedSolver(SAT_Answer(False, None)),
        ]
    actions     = [SAT_WorldAction(s, TimeDelta(seconds = 16.0)) for s in subsolvers]
    environment = Environment()

    # each test is similar
    def test_now(seconds, answer, clean_record):
        """
        Test the SAT solver portfolio shell.
        """

        import numpy

        from borg.tasks import FileTask

        strategy = SequenceSelectionStrategy(actions)
        solver   = PortfolioSolver(strategy)
        task     = FileTask("/tmp/arbitrary_path.cnf")
        attempt  = solver.solve(task, TimeDelta(seconds = seconds), numpy.random, environment)

        assert_equal(attempt.answer, answer)
        assert_equal([s for (s, _) in attempt.record], clean_record)

    # yield the individual tests
    yield (test_now,  2.0, None,                          [])
    yield (test_now, 18.0, None,                          subsolvers[:1])
    yield (test_now, 34.0, SAT_Answer(True, certificate), subsolvers[:2])
    yield (test_now, 72.0, SAT_Answer(True, certificate), subsolvers[:2])

