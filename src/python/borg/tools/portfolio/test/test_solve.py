"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_tools_portfolio_solve():
    """
    Test the portfolio solver execution script.
    """

    # build a solver
    from tempfile                  import NamedTemporaryFile
    from borg.sat                  import Decision
    from borg.solvers.test.support import FixedSolver

    fixed_solver = FixedSolver(Decision(True, [1, 2, 3, 4, 0]))

    with NamedTemporaryFile(suffix = ".pickle") as pickle_file:
        # write it to disk
        import cPickle as pickle

        pickle.dump(fixed_solver, pickle_file, -1)

        pickle_file.flush()

        # prepare to invoke the "solve" script
        from borg.tools.portfolio.test.support import clean_up_environment

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

            from cargo.temporal import TimeDelta
            from borg.tasks     import FileTask
            from borg.solvers   import (
                Environment,
                SAT_CompetitionSolver,
                )

            script_solver = \
                SAT_CompetitionSolver(
                    command = [
                        "python",
                        "-m",
                        "borg.tools.portfolio.solve",
                        pickle_file.name,
                        cnf_file.name,
                        "42",
                        ],
                    )
            task          = FileTask(cnf_file.name)
            environment   = Environment()
            budget        = TimeDelta(seconds = 16.0)
            attempt       = script_solver.solve(task, budget, numpy.random, environment)

    # does the result match expectations?
    assert_equal(attempt.answer, fixed_solver.answer)

