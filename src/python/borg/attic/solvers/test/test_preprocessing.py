"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_preprocessing_solver():
    """
    Test the preprocessing solver.
    """

    # set up the solver
    from datetime                  import timedelta
    from borg.sat                  import Decision
    from borg.tasks                import Task
    from borg.solvers              import (
        Environment,
        PreprocessingSolver,
        )
    from borg.solvers.test.support import (
        FixedSolver,
        FixedPreprocessor,
        )

    task               = Task()
    fixed_solver       = FixedSolver(Decision(True, [1, 2, 3, 4, 0]))
    fixed_preprocessor = FixedPreprocessor(False, None, timedelta(seconds = 8.0))
    environment        = Environment()
    solver             = PreprocessingSolver(fixed_preprocessor, fixed_solver)

    # test it
    from nose.tools import assert_equal

    attempt = solver.solve(task, timedelta(seconds = 7.0), None, environment)

    assert_equal(attempt.answer, None)

    attempt = solver.solve(task, timedelta(seconds = 15.0), None, environment)

    assert_equal(attempt.answer, fixed_solver.answer)

