"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    timed,
    assert_equal,
    )

def test_sat_competition_solver():
    """
    Test the SatELite-interface preprocessor.
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
    @timed(32.0)
    def test_satelite(solver_code, preprocesses, answer):
        """
        Test the SatELite-interface preprocessor.
        """

        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix = ".py") as code_file:
            # write the fake preprocessor program
            code_file.write(solver_code)
            code_file.flush()

            # build the preprocessor
            from cargo.temporal           import TimeDelta
            from utexas.sat.tasks         import FileTask
            from utexas.sat.preprocessors import SatELitePreprocessor

            satelite = SatELitePreprocessor(["python", code_file.name])

            # run the preprocessor
            task   = FileTask("/tmp/path_irrelevant.cnf")
            budget = TimeDelta(seconds = 8.0)
            result = satelite.preprocess(task, budget, "/tmp/arbitrary", None, None)

            # verify its response
            assert_equal(result.answer, answer)

            if preprocesses:
                assert result.output_task is not result.input_task
            else:
                assert result.output_task is result.input_task

    # run each test
    from utexas.sat import SAT_Answer

    yield (test_satelite, finds_sat_code,    False, SAT_Answer(True, [42, 0]))
    yield (test_satelite, finds_unsat_code,  False, SAT_Answer(False))
    yield (test_satelite, preprocesses_code, True,  None)
    yield (test_satelite, fails_code,        False, None)
    yield (test_satelite, times_out_code,    False, None)

