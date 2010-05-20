"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_lookup_solver():
    """
    Test the lookup-based solver.
    """

    # set up the solver
    from borg.sat                  import SAT_Answer
    from borg.solvers              import (
        Environment,
        LookupSolver,
        )
    from borg.solvers.test.support import FixedSolver

    foo_solver    = FixedSolver(SAT_Answer(True, [1, 2, 3, 4, 0]))
    environment   = Environment(named_solvers = {"foo": foo_solver})
    solver        = LookupSolver("foo")

    # test it
    from nose.tools import assert_equal

    attempt = solver.solve(None, None, None, environment)

    assert_equal(attempt.answer, foo_solver.answer)

def test_lookup_preprocessor():
    """
    Test the lookup-based preprocessor.
    """

    # the body of each
    def test_preprocessor(input_task, preprocess, answer):
        """
        Test the lookup-based preprocessor.
        """

        import numpy

        from cargo.temporal            import TimeDelta
        from borg.solvers              import (
            Environment,
            LookupPreprocessor,
            )
        from borg.solvers.test.support import FixedPreprocessor

        named       = FixedPreprocessor(preprocess, answer)
        environment = Environment(named_solvers = {"baz": named})
        lookup      = LookupPreprocessor("baz")
        result      = \
            lookup.preprocess(
                input_task,
                TimeDelta(seconds = 32.0),
                "/tmp/arbitrary_directory",
                numpy.random,
                environment,
                )

        # assert our expectations
        from nose.tools import (
            assert_true,
            assert_equal,
            )

        assert_equal(result.answer, answer)

        if preprocess:
            assert_true(result.task != result.output_task)
        else:
            assert_true(result.task == result.output_task)

    # each test
    from borg.sat   import SAT_Answer
    from borg.tasks import (
        FileTask,
        PreprocessedTask,
        )

    input_task = FileTask("/tmp/arbitrary_path.cnf")
    answer     = SAT_Answer(True, [42])

    yield (test_preprocessor, input_task, False, answer)
    yield (test_preprocessor, input_task, False, None)
    yield (test_preprocessor, input_task, True,  None)

