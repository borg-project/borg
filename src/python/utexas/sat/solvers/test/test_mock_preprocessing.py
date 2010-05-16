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

    # populate the test database
    from utexas.sat.solvers.test.support import FakeSolverData

    fake_data = FakeSolverData()

    @with_setup(fake_data.set_up, fake_data.tear_down)
    def test_solver(preprocessor_name, solver_name, task_uuid, seconds, satisfiable):
        """
        Test SAT_MockCompetitionSolver behavior.
        """

        from utexas.data             import TaskRow
        from utexas.sat.tasks        import SAT_MockTask
        from utexas.sat.solvers      import (
            SAT_Environment,
            SAT_MockCompetitionSolver,
            SAT_MockPreprocessingSolver,
            )
        from cargo.temporal          import TimeDelta

        task         = SAT_MockTask(task_uuid)
        inner_solver = SAT_MockCompetitionSolver(solver_name)
        outer_solver = SAT_MockPreprocessingSolver(preprocessor_name, inner_solver)
        environment  = SAT_Environment(CacheSession = fake_data.Session)
        result       = outer_solver.solve(task, TimeDelta(seconds = seconds), None, environment)

        assert_equal(result.satisfiable, satisfiable)
        assert_equal(result.certificate, None)

    # test the bar-foo combination on fake data
    from utexas.sat.solvers.test.support import task_uuids

    yield (test_solver, "bar", "foo", task_uuids[0],  7.0, None)
    yield (test_solver, "bar", "foo", task_uuids[0],  9.0, None)
    yield (test_solver, "bar", "foo", task_uuids[0], 17.0, True)
    yield (test_solver, "bar", "foo", task_uuids[1],  7.0, None)
    yield (test_solver, "bar", "foo", task_uuids[1],  9.0, False)
    yield (test_solver, "bar", "foo", task_uuids[2],  7.0, None)
    yield (test_solver, "bar", "foo", task_uuids[2],  9.0, True)

    # test the baz-fob combination on fake data
    yield (test_solver, "baz", "fob", task_uuids[0],  7.0, None)
    yield (test_solver, "baz", "fob", task_uuids[0],  9.0, None)
    yield (test_solver, "baz", "fob", task_uuids[0], 17.0, True)
    yield (test_solver, "baz", "fob", task_uuids[1],  7.0, None)
    yield (test_solver, "baz", "fob", task_uuids[1],  9.0, None)
    yield (test_solver, "baz", "fob", task_uuids[1], 17.0, False)
    yield (test_solver, "baz", "fob", task_uuids[2], 17.0, None)

