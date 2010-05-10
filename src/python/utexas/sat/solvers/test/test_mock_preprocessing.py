"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    with_setup,
    assert_equal,
    )

def test_mock_preprocessing_simple():
    """
    Test SAT_MockPreprocessingSolver behavior.
    """

    from utexas.data                       import SAT_TaskRow
    from utexas.sat.tasks                  import SAT_MockFileTask
    from utexas.sat.solvers                import (
        SAT_MockCompetitionSolver,
        SAT_MockPreprocessingSolver,
        )
    from utexas.sat.solvers.test.fake_data import (
        FakeSolverData,
        task_uuids,
        )
    from cargo.temporal                    import TimeDelta

    fake_data = FakeSolverData()

    @with_setup(fake_data.set_up, fake_data.tear_down)
    def test_solver(task_uuid, seconds, satisfiable, certificate):
        """
        Test SAT_MockCompetitionSolver behavior.
        """

        task_row     = fake_data.session.query(SAT_TaskRow).get(task_uuid)
        task         = SAT_MockFileTask(task_row)
        inner_solver = SAT_MockCompetitionSolver("foo_solver", fake_data.engine)
        outer_solver = SAT_MockPreprocessingSolver("bar_preprocessor", inner_solver, fake_data.engine)
        result       = outer_solver.solve(task, cutoff = TimeDelta(seconds = seconds))

        assert_equal(result.satisfiable, satisfiable)
        assert_equal(result.certificate, certificate)

    # test on several fake tasks
    yield (test_solver, task_uuids[0], 24.0, True,  [43])
    yield (test_solver, task_uuids[1], 24.0, False, None)
    yield (test_solver, task_uuids[1],  4.0, None,  None)
    yield (test_solver, task_uuids[2], 24.0, False, None)
    yield (test_solver, task_uuids[3], 24.0, True,  [43])
    yield (test_solver, task_uuids[3],  4.0, None,  None)
    yield (test_solver, task_uuids[4], 24.0, True,  [43])

