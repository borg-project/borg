"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_satelite_preprocessor_preprocess():
    """
    Test the preprocessing interface to SatELite.
    """

    # define various fake versions of SatELite
    finds_sat_code    = \
"""
print "c comment"
print "s SATISFIABLE"
print "v 42 0"
raise SystemExit(10)
"""
    finds_unsat_code  = \
"""
print "c comment"
print "s UNSATISFIABLE"
raise SystemExit(20)
"""
    preprocesses_code = \
"""
raise SystemExit(0)
"""
    fails_code        = \
"""
raise SystemExit(1)
"""
    times_out_code    = \
"""
print "c comment"
while True: pass
"""

    # the body of each test
    from nose.tools import timed

    @timed(32.0)
    def test_satelite(solver_code, preprocesses, answer):
        """
        Test the preprocessing interface to SatELite.
        """

        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix = ".py") as code_file:
            # write the fake preprocessor program
            code_file.write(solver_code)
            code_file.flush()

            # build the preprocessor
            from borg.solvers import SatELitePreprocessor

            satelite = SatELitePreprocessor(["python", code_file.name])

            # run the preprocessor
            from cargo.temporal import TimeDelta
            from borg.tasks     import FileTask

            task    = FileTask("/tmp/path_irrelevant.cnf")
            budget  = TimeDelta(seconds = 8.0)
            attempt = satelite.preprocess(task, budget, "/tmp/arbitrary", None, None)

            # verify its response
            from nose.tools import (
                assert_true,
                assert_equal,
                )

            assert_equal(attempt.answer, answer)

            if preprocesses:
                assert_true(attempt.output_task is not attempt.task)
            else:
                assert_true(attempt.output_task is attempt.task)

    # run each test
    from borg.sat import Decision

    yield (test_satelite, finds_sat_code,    False, Decision(True, [42, 0]))
    yield (test_satelite, finds_unsat_code,  False, Decision(False))
    yield (test_satelite, preprocesses_code, True,  None)
    yield (test_satelite, fails_code,        False, None)
    yield (test_satelite, times_out_code,    False, None)

def test_satelite_preprocessor_extend():
    """
    Test the model extension interface to SatELite.
    """

    extends_model_code = \
"""
print "v 1 2 3 4 0"
raise SystemExit(10)
"""

    # the body of each test
    from nose.tools import timed

    @timed(32.0)
    def test_satelite(solver_code, answer_in, answer_out):
        """
        Test the model extension interface to SatELite.
        """

        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix = ".py") as code_file:
            # write the fake preprocessor program
            code_file.write(solver_code)
            code_file.flush()

            # build the preprocessor
            from borg.solvers import SatELitePreprocessor

            satelite = SatELitePreprocessor(["python", code_file.name])

            # run the preprocessor
            from borg.tasks import (
                FileTask,
                PreprocessedDirectoryTask,
                )

            task_in  = FileTask("/tmp/path_irrelevant.cnf")
            task     = PreprocessedDirectoryTask(satelite, None, task_in, "/tmp/foo/", "bar.cnf")
            extended = satelite.extend(task, answer_in, None)

            # verify its response
            from nose.tools import assert_equal

            assert_equal(extended, answer_out)

    # run each test
    from borg.sat import Decision

    answer_in  = Decision(True, [42, 0])
    answer_out = Decision(True, [1, 2, 3, 4, 0])

    yield (test_satelite, extends_model_code, answer_in, answer_out)

