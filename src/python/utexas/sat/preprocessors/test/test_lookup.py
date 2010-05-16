"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    assert_true,
    assert_equal,
    )

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

