"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from uuid               import uuid4
from nose.tools         import assert_equal
from utexas.sat.solvers import SAT_Solver

task_uuids      = [uuid4() for i in xrange(3)]
baz_task_uuids  = [uuid4() for i in xrange(3)]
unsanitized_cnf = \
"""c comment
c foo
p cnf 2 6
 -1 2 3 0   
-4 -5   6 0
%
0
"""
sanitized_cnf   = \
"""c comment
c foo
p cnf 2 6
-1 2 3 0
-4 -5 6 0
"""

class TaskVerifyingSolver(SAT_Solver):
    """
    Solver that merely verifies the contents of the task.
    """

    def __init__(self, correct_cnf):
        """
        Initialize.
        """

        self.correct_cnf = correct_cnf

    def solve(self, task, budget, random, environment):
        """
        Verify behavior.
        """

        from utexas.sat.solvers import SAT_BareResult

        with open(task.path) as task_file:
            assert_equal(task_file.read(), self.correct_cnf)

        return SAT_BareResult(self, task, budget, budget, None, None)

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
    baz_task_rows = [
        PTR(uuid = u, input_task = t, preprocessor_name = "baz")
        for (u, t) in zip(baz_task_uuids, task_rows)
        ]

    # behavior of foo on raw tasks
    add_solver_run("foo", task_rows[0], True)
    add_solver_run("foo", task_rows[1], False)
    add_solver_run("foo", task_rows[2], None)

    # behavior of fob on raw tasks
    add_solver_run("fob", task_rows[0], None)
    add_solver_run("fob", task_rows[1], None)
    add_solver_run("fob", task_rows[2], None)

    # bar yields no preprocessed tasks
    add_preprocessor_run("bar", task_rows[0], task_rows[0], None)
    add_preprocessor_run("bar", task_rows[1], task_rows[1], False)
    add_preprocessor_run("bar", task_rows[2], task_rows[2], True)

    # baz successfully preprocesses tasks
    add_preprocessor_run("baz", task_rows[0], baz_task_rows[0], None)
    add_preprocessor_run("baz", task_rows[1], baz_task_rows[1], None)
    add_preprocessor_run("baz", task_rows[2], baz_task_rows[2], None)

    # behavior of fob on preprocessed tasks
    add_solver_run("fob", baz_task_rows[0], True)
    add_solver_run("fob", baz_task_rows[1], False)
    add_solver_run("fob", baz_task_rows[2], None)

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

        self.engine  = create_engine("sqlite:///:memory:", echo = False)
        self.Session = make_session(bind = self.engine)
        self.session = self.Session()

        add_fake_runs(self.session)

    def tear_down(self):
        """
        Clean up after a test.
        """

        self.session.close()
        self.engine.dispose()

