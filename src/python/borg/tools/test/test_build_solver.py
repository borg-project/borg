"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_build_solver():
    """
    Test the solver construction tool.
    """

    # fork; build a random-model solver
    from os.path    import join
    from cargo.io   import mkdtemp_scoped
    from cargo.log  import get_logger
    from cargo.json import save_json

    log = get_logger(__name__, level = "NOTSET")

    with mkdtemp_scoped() as sandbox_path:
        # invoke the script
        from cargo.io import (
            unset_all,
            call_capturing,
            )
        from borg     import get_support_path

        uuids_json_path    = join(sandbox_path, "train_uuids.json")
        solver_pickle_path = join(sandbox_path, "solver.pickle")

        save_json([], uuids_json_path)

        from borg import get_support_path

        (stdout, stderr, code) = \
            call_capturing(
                [
                    "python",
                    "-m",
                    "borg.tools.build_solver",
                    uuids_json_path,
                    get_support_path("for_tests/foo_solver.py"),
                    solver_pickle_path,
                    ],
                preexec_fn = lambda: unset_all("CARGO_FLAGS_EXTRA_FILE"),
                )

        log.debug("call stdout follows:\n%s", stdout)
        log.debug("call stderr follows:\n%s", stderr)

        assert_equal(code, 0)

        # and load the solver file
        import cPickle as pickle

        with open(solver_pickle_path) as file:
            solver = pickle.load(file)

    # execute it in an environment with appropriate fake solvers mapped
    import numpy

    from datetime                  import timedelta
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
    attempt       = solver.solve(task, timedelta(seconds = 1e6), numpy.random, environment)

    assert_equal(attempt.answer, fixed_solver.answer)

