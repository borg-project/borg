"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools         import assert_equal
from utexas.sat.solvers import SAT_Solver

class FixedSolver(SAT_Solver):
    """
    A fake, fixed-result solver.
    """

    def __init__(self, satisfiable, certificate, record):
        """
        Initialize.
        """

        self.satisfiable = satisfiable
        self.certificate = certificate
        self.record      = record

    def solve(self, task, budget, random, environment):
        """
        Pretend to solve the task.
        """

        from utexas.sat.solvers import SAT_BareResult

        self.record.append(self)

        return \
            SAT_BareResult(
                self,
                task,
                budget,
                budget,
                self.satisfiable,
                self.certificate,
                )

def test_sat_portfolio_solver():
    """
    Test the SAT solver portfolio shell.
    """

    # set up the portfolio solver
    from cargo.temporal              import TimeDelta
    from utexas.sat.solvers          import SAT_PortfolioSolver
    from utexas.portfolio.sat_world  import SAT_WorldAction
    from utexas.portfolio.strategies import SequenceSelectionStrategy

    record      = []
    certificate = [1, 2, 3, 4, 0]
    subsolvers  = [
        FixedSolver(None,  None,        record),
        FixedSolver(True,  certificate, record),
        FixedSolver(False, None,        record),
        ]
    actions     = [SAT_WorldAction(s, TimeDelta(seconds = 16.0)) for s in subsolvers]

    # each test is similar
    def test_now(seconds, satisfiable, certificate, clean_record):
        """
        Test the SAT solver portfolio shell.
        """

        import numpy

        from utexas.sat.tasks import SAT_FileTask

        del record[:]

        strategy = SequenceSelectionStrategy(actions)
        solver   = SAT_PortfolioSolver(strategy)
        task     = SAT_FileTask("/tmp/arbitrary_path.cnf")
        result   = solver.solve(task, TimeDelta(seconds = seconds), numpy.random, None)

        assert_equal(result.satisfiable, satisfiable)
        assert_equal(result.certificate, certificate)
        assert_equal(record,             clean_record)

    # yield the individual tests
    yield (test_now,  2.0, None, None,        [])
    yield (test_now, 18.0, None, None,        subsolvers[:1])
    yield (test_now, 34.0, True, certificate, subsolvers[:2])
    yield (test_now, 72.0, True, certificate, subsolvers[:2])

