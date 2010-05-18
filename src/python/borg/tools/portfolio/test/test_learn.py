"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_tools_portfolio_learn():
    """
    Test the portfolio solver construction script.
    """

    # fork; build a random-model solver
    from os.path    import join
    from cargo.io   import mkdtemp_scoped

    with mkdtemp_scoped() as sandbox_path:
        # invoke the script
        from subprocess                          import check_call
        from utexas.tools.portfolio.test.support import clean_up_environment

        solver_pickle_path = join(sandbox_path, "solver.pickle")

        with open("/dev/null", "w") as null_file:
            check_call(
                [
                    "python",
                    "-m",
                    "utexas.tools.portfolio.learn",
                    "-m",
                    "random",
                    "none",
                    solver_pickle_path,
                    ],
                stdout     = null_file,
                stderr     = null_file,
                preexec_fn = clean_up_environment,
                )

        # and load the solver file
        import cPickle as pickle

        with open(solver_pickle_path) as file:
            solver = pickle.load(file)

    # execute it in an environment with appropriate fake solvers mapped
    import numpy

    from cargo.temporal                  import TimeDelta
    from utexas.sat.tasks                import FileTask
    from utexas.sat.solvers              import SAT_Environment
    from utexas.sat.solvers.test.support import FixedSolver

    fixed_solver  = FixedSolver(True, [1, 2, 3, 4, 0])
    named_solvers = {
        "sat/2009/CirCUs"         : fixed_solver,
        "sat/2009/clasp"          : fixed_solver,
        "sat/2009/glucose"        : fixed_solver,
        "sat/2009/LySAT_i"        : fixed_solver,
        "sat/2009/minisat_09z"    : fixed_solver,
        "sat/2009/minisat_cumr_p" : fixed_solver,
        "sat/2009/mxc_09"         : fixed_solver,
        "sat/2009/precosat"       : fixed_solver,
        "sat/2009/rsat_09"        : fixed_solver,
        "sat/2009/SApperloT"      : fixed_solver,
        }
    environment   = SAT_Environment(named_solvers = named_solvers)
    task          = FileTask("/tmp/arbitrary_path.cnf")
    result        = solver.solve(task, TimeDelta(seconds = 1e6), numpy.random, environment)

    assert_equal(result.satisfiable, fixed_solver.satisfiable)
    assert_equal(result.certificate, fixed_solver.certificate)

