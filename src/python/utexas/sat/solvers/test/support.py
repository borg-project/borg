"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from uuid               import uuid4
from utexas.sat.solvers import SAT_Solver

task_uuids = [uuid4() for i in xrange(5)]

class FixedSolver(SAT_Solver):
    """
    A fake, fixed-result solver.
    """

    def __init__(self, satisfiable, certificate):
        """
        Initialize.
        """

        self.satisfiable = satisfiable
        self.certificate = certificate

    def solve(self, task, budget, random, environment):
        """
        Pretend to solve the task.
        """

        from utexas.sat.solvers import SAT_BareResult

        return \
            SAT_BareResult(
                self,
                task,
                budget,
                budget,
                self.satisfiable,
                self.certificate,
                )

def add_fake_runs(session):
    """
    Insert standard test data into an empty database.
    """

    from utexas.data import (
        DatumBase,
        SAT_TaskRow,
        SAT_TrialRow,
        SAT_AnswerRow,
        CPU_LimitedRunRow,
        SAT_RunAttemptRow,
        PreprocessorRunRow,
        PreprocessedTaskRow,
        )

    # layout
    DatumBase.metadata.create_all(session.connection().engine)

    # add the recyclable-run trial
    recyclable_trial_row = SAT_TrialRow(uuid = SAT_TrialRow.RECYCLABLE_UUID)

    session.add(recyclable_trial_row)

    # prepare to insert solver runs
    def add_solver_run(solver_name, task_row, satisfiable):
        """
        Insert a fake run.
        """

        certificate = [42] if satisfiable else None
        attempt_row = \
            SAT_RunAttemptRow(
                task        = task_row,
                answer      = SAT_AnswerRow(satisfiable, certificate),
                budget      = 32.0,
                cost        = 8.0,
                trials      = [recyclable_trial_row],
                run         = CPU_LimitedRunRow(proc_elapsed = 8.0, cutoff = 32.0),
                solver_name = solver_name,
                )

        session.add(attempt_row)

    # prepare to insert preprocessor runs
    def add_preprocessor_run(preprocessor_name, input_task_row, output_task_row, satisfiable):
        """
        Insert a fake run.
        """

        if satisfiable is True:
            answer = SAT_AnswerRow(True, [42])
        elif satisfiable is False:
            answer = SAT_AnswerRow(False)
        else:
            answer = None

        run_row = \
            PreprocessorRunRow(
                preprocessor_name = preprocessor_name,
                input_task        = input_task_row,
                output_task       = output_task_row,
                run               = CPU_LimitedRunRow(proc_elapsed = 8.0, cutoff = 32.0),
                answer            = answer,
                budget            = 32.0,
                cost              = 8.0,
                trials            = [recyclable_trial_row],
                )

        session.add(run_row)

    # add the tasks
    PTR = PreprocessedTaskRow

    task_rows     = [SAT_TaskRow(uuid = u) for u in task_uuids]
    baz_task_rows = [PTR(input_task = t, preprocessor_name = "baz") for t in task_rows]

    # add fake runs
    add_solver_run("foo", task_rows[0], True)
    add_solver_run("foo", task_rows[1], False)
    add_solver_run("foo", task_rows[2], None)

    add_solver_run("fob", task_rows[0], None)
    add_solver_run("fob", task_rows[1], None)
    add_solver_run("fob", task_rows[2], None)

    add_preprocessor_run("bar", task_rows[0], task_rows[0], None)
    add_preprocessor_run("bar", task_rows[1], task_rows[1], True)
    add_preprocessor_run("bar", task_rows[2], task_rows[2], False)

    add_preprocessor_run("baz", task_rows[0], baz_task_rows[0], None)
    add_preprocessor_run("baz", task_rows[1], baz_task_rows[1], None)
    add_preprocessor_run("baz", task_rows[2], baz_task_rows[2], None)

    add_solver_run("fob", baz_task_rows[0], True)
    add_solver_run("fob", baz_task_rows[1], False)
    add_solver_run("fob", baz_task_rows[2], True)

    session.commit()

class FakeSolverData(object):
    """
    Tests of the mock solver(s).
    """

    def set_up(self):
        """
        Prepare for a test.
        """

        from sqlalchemy        import create_engine
        from cargo.sql.alchemy import make_session

        self.engine  = create_engine("sqlite:///:memory:")
        self.Session = make_session(bind = self.engine)
        self.session = self.Session()

        add_fake_runs(self.session)

    def tear_down(self):
        """
        Clean up after a test.
        """

        self.session.close()
        self.engine.dispose()

