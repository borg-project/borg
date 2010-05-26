"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from uuid         import uuid4
from nose.tools   import assert_equal
from borg.rowed   import Rowed
from borg.solvers import (
    AbstractSolver,
    AbstractPreprocessor,
    )

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
"""p cnf 2 6
-1 2 3 0
-4 -5 6 0
"""

class TaskVerifyingSolver(Rowed, AbstractSolver):
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

        from borg.solvers import Attempt

        with open(task.path) as task_file:
            assert_equal(task_file.read(), self.correct_cnf)

        return Attempt(self, budget, budget, task, None)

class FixedSolver(Rowed, AbstractSolver):
    """
    A fake, fixed-result solver.
    """

    def __init__(self, answer):
        """
        Initialize.
        """

        self.answer = answer

    def solve(self, task, budget, random, environment):
        """
        Pretend to solve the task.
        """

        from borg.solvers import Attempt

        return \
            Attempt(
                self,
                budget,
                budget,
                task,
                self.answer,
                )

class TaskVerifyingPreprocessor(Rowed, AbstractPreprocessor):
    """
    Preprocessor that merely verifies the contents of the task.
    """

    def __init__(self, correct_cnf):
        """
        Initialize.
        """

        self._correct_cnf = correct_cnf

    def preprocess(self, task, budget, output_dir, random, environment):
        """
        Verify behavior.
        """

        from cargo.unix.accounting import CPU_LimitedRun
        from borg.solvers          import PreprocessorAttempt

        with open(task.path) as task_file:
            assert_equal(task_file.read(), self._correct_cnf)

        return \
            PreprocessorAttempt(
                self,
                task,
                None,
                None,
                CPU_LimitedRun(None, budget, None, None, None, budget, None, None),
                task,
                )

    def solve(): pass
    def extend(): pass
    def make_task(): pass

class FixedPreprocessor(Rowed, AbstractPreprocessor):
    """
    A fake, fixed-result preprocessor.
    """

    def __init__(self, preprocess, answer, cost = None):
        """
        Initialize.
        """

        self._preprocess = preprocess
        self._answer     = answer
        self._cost       = cost

    def preprocess(self, task, budget, output_dir, random, environment):
        """
        Pretend to preprocess an instance.
        """

        from cargo.unix.accounting import CPU_LimitedRun
        from borg.solvers          import PreprocessorAttempt

        if self._cost is None:
            cost = budget
        else:
            cost = self._cost

        if self._preprocess:
            from borg.tasks import PreprocessedTask

            output_task = PreprocessedTask(self, None, task)
        else:
            output_task = task

        return \
            PreprocessorAttempt(
                self,
                task,
                self._answer,
                None,
                CPU_LimitedRun(None, budget, None, None, None, cost, None, None),
                output_task,
                )

    def solve(): pass
    def extend(): pass
    def make_task(): pass

def add_fake_runs(session):
    """
    Insert standard test data into an empty database.
    """

    from borg.data import (
        TaskRow,
        TrialRow,
        DatumBase,
        DecisionRow,
        RunAttemptRow,
        CPU_LimitedRunRow,
        PreprocessedTaskRow,
        PreprocessorAttemptRow,
        )

    # layout
    DatumBase.metadata.create_all(session.connection().engine)

    # add the recyclable-run trial
    recyclable_trial_row = TrialRow(uuid = TrialRow.RECYCLABLE_UUID)

    session.add(recyclable_trial_row)

    # prepare to insert solver runs
    def add_solver_run(solver_name, task_row, satisfiable):
        """
        Insert a fake run.
        """

        if satisfiable is None:
            answer = None
        else:
            answer = DecisionRow(satisfiable, [42] if satisfiable else None)

        attempt_row = \
            RunAttemptRow(
                task        = task_row,
                answer      = answer,
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

        if satisfiable is None:
            answer = None
        else:
            answer = DecisionRow(satisfiable, [42] if satisfiable else None)

        run_row = \
            PreprocessorAttemptRow(
                solver_name = preprocessor_name,
                task        = input_task_row,
                output_task = output_task_row,
                run         = CPU_LimitedRunRow(proc_elapsed = 8.0, cutoff = 32.0),
                answer      = answer,
                budget      = 32.0,
                cost        = 8.0,
                trials      = [recyclable_trial_row],
                )

        session.add(run_row)

    # add the tasks
    PTR = PreprocessedTaskRow

    task_rows     = [TaskRow(uuid = u) for u in task_uuids]
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

