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
    from cargo.log  import get_logger
    from cargo.json import save_json

    log = get_logger(__name__, level = "NOTSET")

    solver_request = {
        "domain"   : "sat",
        "type"     : "portfolio",
        "strategy" : {
            "type"    : "modeling",
            "planner" : {
                "type"     : "hard_myopic",
                "discount" : 1.0,
                },
            "model"   : {
                "type"        : "random",
                "actions"     : {
                    "solvers" : [
                        "foo",
                        "bar",
                        "baz",
                        ],
                    "budgets" : [1],
                    },
                },
            },
        }

    with mkdtemp_scoped() as sandbox_path:
        # invoke the script
        from tempfile                          import TemporaryFile
        from subprocess                        import call
        from borg.tools.portfolio.test.support import clean_up_environment

        uuids_json_path    = join(sandbox_path, "train_uuids.json")
        solver_json_path   = join(sandbox_path, "solver.json")
        solver_pickle_path = join(sandbox_path, "solver.pickle")

        save_json([], uuids_json_path)
        save_json(solver_request, solver_json_path)

        with TemporaryFile() as temporary:
            code = \
                call(
                    [
                        "python",
                        "-m",
                        "borg.tools.portfolio.learn",
                        uuids_json_path,
                        solver_json_path,
                        solver_pickle_path,
                        ],
                    stdout     = temporary,
                    stderr     = temporary,
                    preexec_fn = clean_up_environment,
                    )

            temporary.seek(0)

            log.debug("call output follows:\n%s", temporary.read())

            assert_equal(code, 0)

        # and load the solver file
        import cPickle as pickle

        with open(solver_pickle_path) as file:
            solver = pickle.load(file)

    # execute it in an environment with appropriate fake solvers mapped
    import numpy

    from cargo.temporal            import TimeDelta
    from borg.tasks                import FileTask
    from borg.sat                  import Decision
    from borg.solvers              import Environment
    from borg.solvers.test.support import FixedSolver

    fixed_solver  = FixedSolver(Decision(True, [1, 2, 3, 4, 0]))
    named_solvers = {
        "foo" : fixed_solver,
        "bar" : fixed_solver,
        "baz" : fixed_solver,
        }
    environment   = Environment(named_solvers = named_solvers)
    task          = FileTask("/tmp/arbitrary_path.cnf")
    attempt       = solver.solve(task, TimeDelta(seconds = 1e6), numpy.random, environment)

    assert_equal(attempt.answer, fixed_solver.answer)

