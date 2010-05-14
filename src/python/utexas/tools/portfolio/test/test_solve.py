"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_tools_portfolio_solve():
    """
    Test the portfolio solver execution script.
    """

    # build a solver
    from tempfile                        import NamedTemporaryFile
    from utexas.sat.solvers.test.support import FixedSolver

    original_solver = FixedSolver(True, [1, 2, 3, 4, 0])

    with NamedTemporaryFile(suffix = ".pickle") as pickle_file:
        # write it to disk
        import cPickle as pickle

        pickle.dump(original_solver, pickle_file, -1)

        pickle_file.flush()

        # prepare to invoke the "solve" script
        from utexas.tools.portfolio.test.support import clean_up_environment

        clean_up_environment() # suboptimal, since we modify our own environment; whatever

        with NamedTemporaryFile(suffix = ".cnf") as cnf_file:
            # write a SAT instance to disk
            cnf_file.write(
"""p cnf 2 6
-1 2 3 0
-4 -5 6 0
""",
                )
            cnf_file.flush()

            # invoke the script solver
            import numpy

            from cargo.temporal     import TimeDelta
            from utexas.sat.tasks   import SAT_FileTask
            from utexas.sat.solvers import (
                SAT_Environment,
                SAT_CompetitionSolver,
                )

            script_solver = \
                SAT_CompetitionSolver(
                    command = [
                        "python",
                        "-m",
                        "utexas.tools.portfolio.solve",
                        pickle_file.name,
                        cnf_file.name,
                        "42",
                        ],
                    )
            task          = SAT_FileTask(cnf_file.name)
            environment   = SAT_Environment()
            budget        = TimeDelta(seconds = 16.0)
            result        = script_solver.solve(task, budget, numpy.random, environment)

    # does the result match expectations?
    assert_equal(result.satisfiable, original_solver.satisfiable)
    assert_equal(result.certificate, original_solver.certificate)

