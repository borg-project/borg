"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def generated_callable(fake_data, solver, task_uuid, seconds, answer, output_task_uuid = None):
    """
    Test the recycling solver.
    """

    from nose.tools     import assert_equal
    from cargo.temporal import TimeDelta
    from borg.data      import TaskRow
    from borg.tasks     import Task
    from borg.solvers   import Environment

    task        = Task(TaskRow(uuid = task_uuid))
    environment = Environment(CacheSession = fake_data.Session)

    if output_task_uuid is None:
        attempt = solver.solve(task, TimeDelta(seconds = seconds), None, environment)
    else:
        attempt = solver.preprocess(task, TimeDelta(seconds = seconds), None, None, environment)

        with fake_data.Session() as session:
            output_task_row = attempt.output_task.get_row(session)

            assert_equal(output_task_row.uuid, output_task_uuid)

    assert_equal(attempt.answer, answer)

def test_recycling_solver():
    """
    Test the recycling solver.
    """

    # build the yielded test callable
    from functools                 import partial
    from nose.tools                import with_setup
    from borg.solvers.test.support import FakeSolverData

    fake_data   = FakeSolverData()
    generated   = partial(generated_callable, fake_data)
    test_solver = with_setup(fake_data.set_up, fake_data.tear_down)(generated)

    # test behavior of foo on raw tasks
    from borg.sat                  import SAT_Answer
    from borg.solvers              import RecyclingSolver
    from borg.solvers.test.support import task_uuids

    foo_solver = RecyclingSolver("foo")

    yield (test_solver, foo_solver, task_uuids[0], 9.0, SAT_Answer(True,  [42]))
    yield (test_solver, foo_solver, task_uuids[0], 7.0, None)
    yield (test_solver, foo_solver, task_uuids[1], 9.0, SAT_Answer(False))
    yield (test_solver, foo_solver, task_uuids[1], 7.0, None)
    yield (test_solver, foo_solver, task_uuids[2], 9.0, None)

    # test behavior of fob on preprocessed tasks
    from borg.solvers.test.support import baz_task_uuids

    fob_solver = RecyclingSolver("fob")

    yield (test_solver, fob_solver, baz_task_uuids[0], 9.0, SAT_Answer(True,  [42]))
    yield (test_solver, fob_solver, baz_task_uuids[0], 7.0, None)
    yield (test_solver, fob_solver, baz_task_uuids[1], 9.0, SAT_Answer(False))
    yield (test_solver, fob_solver, baz_task_uuids[1], 7.0, None)

    # test behavior of bar on raw tasks
    from borg.solvers import RecyclingPreprocessor

    bar_solver = RecyclingPreprocessor("bar")

    yield (test_solver, bar_solver, task_uuids[0], 9.0, None)
    yield (test_solver, bar_solver, task_uuids[1], 9.0, SAT_Answer(False))
    yield (test_solver, bar_solver, task_uuids[1], 7.0, None)
    yield (test_solver, bar_solver, task_uuids[2], 9.0, SAT_Answer(True,  [42]))
    yield (test_solver, bar_solver, task_uuids[2], 7.0, None)

def test_recycling_preprocessor():
    """
    Test the recycling preprocessor.
    """

    # build the yielded test callable
    from functools                 import partial
    from nose.tools                import with_setup
    from borg.solvers.test.support import FakeSolverData

    fake_data   = FakeSolverData()
    generated   = partial(generated_callable, fake_data)
    test_solver = with_setup(fake_data.set_up, fake_data.tear_down)(generated)

#     from nose.tools                import with_setup
#     from borg.solvers.test.support import FakeSolverData

#     fake_data = FakeSolverData()

#     @with_setup(fake_data.set_up, fake_data.tear_down)
#     def test_solver(solver, task_uuid, seconds, answer, output_task_uuid):
#         """
#         Test the recycling solver.
#         """

#         from nose.tools     import assert_equal
#         from cargo.temporal import TimeDelta
#         from borg.data      import TaskRow
#         from borg.tasks     import Task
#         from borg.solvers   import Environment

#         task        = Task(TaskRow(uuid = task_uuid))
#         environment = Environment(CacheSession = fake_data.Session)
#         attempt     = solver.preprocess(task, TimeDelta(seconds = seconds), None, environment)

#         assert_equal(attempt.answer, answer)

#         with fake_data.Session() as session:
#             output_task_row = attempt.output_task.get_row(session)

#             assert_equal(output_task_row.uuid, output_task_uuid)

    # test behavior of the bar preprocessor
    from borg.sat                  import SAT_Answer
    from borg.solvers              import RecyclingPreprocessor
    from borg.solvers.test.support import task_uuids

    bar_solver = RecyclingPreprocessor("bar")

    yield (test_solver, bar_solver, task_uuids[0], 9.0, None,                    task_uuids[0])
    yield (test_solver, bar_solver, task_uuids[1], 9.0, SAT_Answer(False),       task_uuids[1])
    yield (test_solver, bar_solver, task_uuids[1], 7.0, None,                    task_uuids[1])
    yield (test_solver, bar_solver, task_uuids[2], 9.0, SAT_Answer(True,  [42]), task_uuids[2])
    yield (test_solver, bar_solver, task_uuids[2], 7.0, None,                    task_uuids[2])

    # test the behavior of the baz preprocessor
    from borg.solvers.test.support import baz_task_uuids

    baz_solver = RecyclingPreprocessor("baz")

    yield (test_solver, baz_solver, task_uuids[0], 7.0, None,     task_uuids[0])
    yield (test_solver, baz_solver, task_uuids[0], 9.0, None, baz_task_uuids[0])
    yield (test_solver, baz_solver, task_uuids[1], 7.0, None,     task_uuids[1])
    yield (test_solver, baz_solver, task_uuids[1], 9.0, None, baz_task_uuids[1])
    yield (test_solver, baz_solver, task_uuids[2], 7.0, None,     task_uuids[2])
    yield (test_solver, baz_solver, task_uuids[2], 9.0, None, baz_task_uuids[2])

def test_preprocessing_solver_with_recycling():
    """
    Test the preprocessing solver using recycling innards.
    """

    # build the yielded test callable
    from functools                 import partial
    from nose.tools                import with_setup
    from borg.solvers.test.support import FakeSolverData

    fake_data   = FakeSolverData()
    generated   = partial(generated_callable, fake_data)
    test_solver = with_setup(fake_data.set_up, fake_data.tear_down)(generated)

    # test the bar-foo combination on fake data
    from borg.sat                  import SAT_Answer
    from borg.solvers              import (
        RecyclingSolver,
        PreprocessingSolver,
        RecyclingPreprocessor,
        )
    from borg.solvers.test.support import task_uuids

    bar_preprocessor = RecyclingPreprocessor("bar")
    foo_solver       = RecyclingSolver("foo")
    bar_foo_solver   = PreprocessingSolver(bar_preprocessor, foo_solver)

    yield (test_solver, bar_foo_solver, task_uuids[0],  7.0, None)
    yield (test_solver, bar_foo_solver, task_uuids[0],  9.0, None)
    yield (test_solver, bar_foo_solver, task_uuids[0], 17.0, SAT_Answer(True, [42]))
    yield (test_solver, bar_foo_solver, task_uuids[1],  7.0, None)
    yield (test_solver, bar_foo_solver, task_uuids[1],  9.0, SAT_Answer(False))
    yield (test_solver, bar_foo_solver, task_uuids[2],  7.0, None)
    yield (test_solver, bar_foo_solver, task_uuids[2],  9.0, SAT_Answer(True, [42]))

    # test the baz-fob combination on fake data
    baz_preprocessor = RecyclingPreprocessor("baz")
    fob_solver      = RecyclingSolver("fob")
    baz_fob_solver  = PreprocessingSolver(baz_preprocessor, fob_solver)

    yield (test_solver, baz_fob_solver, task_uuids[0],  7.0, None)
    yield (test_solver, baz_fob_solver, task_uuids[0],  9.0, None)
    yield (test_solver, baz_fob_solver, task_uuids[0], 17.0, SAT_Answer(True, [42]))
    yield (test_solver, baz_fob_solver, task_uuids[1],  7.0, None)
    yield (test_solver, baz_fob_solver, task_uuids[1],  9.0, None)
    yield (test_solver, baz_fob_solver, task_uuids[1], 17.0, SAT_Answer(False))
    yield (test_solver, baz_fob_solver, task_uuids[2], 17.0, None)

