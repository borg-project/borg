"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_recycling_solver():
    """
    Test recycling-solver behavior.
    """

    from nose.tools                      import (
        with_setup,
        assert_equal,
        )
    from utexas.data                     import TaskRow
    from utexas.sat.tasks                import MockTask
    from utexas.sat.solvers              import (
        Environment,
        RecyclingSolver,
        )
    from utexas.sat.solvers.test.support import (
        FakeSolverData,
        task_uuids,
        baz_task_uuids,
        )
    from cargo.temporal                  import TimeDelta

    fake_data = FakeSolverData()

    @with_setup(fake_data.set_up, fake_data.tear_down)
    def test_solver(solver_name, task_uuid, seconds, satisfiable, certificate):
        """
        Test recycling-solver behavior.
        """

        task        = MockTask(task_uuid)
        solver      = SAT_MockCompetitionSolver(solver_name)
        environment = SAT_Environment(CacheSession = fake_data.Session)
        result      = solver.solve(task, TimeDelta(seconds = seconds), None, environment)

        assert_equal(result.satisfiable, satisfiable)
        assert_equal(result.certificate, certificate)

    # test behavior of foo on raw tasks
    yield (test_solver, "foo", task_uuids[0], 9.0, True,  [42])
    yield (test_solver, "foo", task_uuids[0], 7.0, None,  None)
    yield (test_solver, "foo", task_uuids[1], 9.0, False, None)
    yield (test_solver, "foo", task_uuids[1], 7.0, None,  None)
    yield (test_solver, "foo", task_uuids[2], 9.0, None,  None)

    # test behavior of fob on preprocessed tasks
    yield (test_solver, "fob", baz_task_uuids[0], 9.0, True,  [42])
    yield (test_solver, "fob", baz_task_uuids[0], 7.0, None,  None)
    yield (test_solver, "fob", baz_task_uuids[1], 9.0, False, None)
    yield (test_solver, "fob", baz_task_uuids[1], 7.0, None,  None)

