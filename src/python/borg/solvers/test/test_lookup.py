"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

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
    attempt = solver.solve(None, None, None, environment)

    assert_equal(attempt.answer, foo_solver.answer)

def test_lookup_preprocessor():
    """
    Test the lookup-based preprocessor.
    """

    # the body of each
    def test_preprocessor(input_task, output_task, answer):
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

        named       = FixedPreprocessor(output_task, answer)
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

        assert_equal(result.output_task, output_task)
        assert_equal(result.answer, answer)

    # each test
    from borg.sat   import SAT_Answer
    from borg.tasks import FileTask

    input_task  = FileTask("/tmp/arbitrary_path.cnf")
    output_task = FileTask("/tmp/arbitrary_path2.cnf")
    answer      = SAT_Answer(True, [42])

    yield (test_preprocessor, input_task, input_task, answer)
    yield (test_preprocessor, input_task, input_task, None)
    yield (test_preprocessor, input_task, output_task, None)

