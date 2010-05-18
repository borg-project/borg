"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools                      import assert_equal
from utexas.sat.solvers.test.support import FixedSolver

def test_named_solver():
    """
    Test the lookup-based SAT solver.
    """

    # set up the solver
    from utexas.sat.solvers import (
        SAT_Environment,
        SAT_LookupSolver,
        )

    certificate   = [1, 2, 3, 4, 0]
    foo_solver    = FixedSolver(True, certificate)
    named_solvers = {
        "foo": foo_solver,
        }
    environment   = SAT_Environment(named_solvers = named_solvers)
    solver        = SAT_LookupSolver("foo")

    # test it
    result = solver.solve(None, None, None, environment)

    assert_equal(result.satisfiable, foo_solver.satisfiable)
    assert_equal(result.certificate, foo_solver.certificate)

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

        from cargo.temporal                        import TimeDelta
        from utexas.sat.solvers                    import SAT_Environment
        from utexas.sat.preprocessors              import LookupPreprocessor
        from utexas.sat.preprocessors.test.support import FixedPreprocessor

        named       = FixedPreprocessor(output_task, answer)
        environment = SAT_Environment(named_preprocessors    = {"baz": named})
        lookup      = LookupPreprocessor("baz")
        output_dir  = "/tmp/arbitrary_directory"
        result      = \
            lookup.preprocess(
                input_task,
                TimeDelta(seconds = 32.0),
                output_dir,
                numpy.random,
                environment,
                )

        assert_equal(result.output_task, output_task)
        assert_equal(result.answer, answer)

    # each test
    from utexas.sat import (
        FileTask,
        SAT_Answer,
        )

    input_task  = FileTask("/tmp/arbitrary_path.cnf")
    output_task = FileTask("/tmp/arbitrary_path2.cnf")
    answer      = SAT_Answer(True, [42])

    yield (test_preprocessor, input_task, input_task, answer)
    yield (test_preprocessor, input_task, input_task, None)
    yield (test_preprocessor, input_task, output_task, None)

