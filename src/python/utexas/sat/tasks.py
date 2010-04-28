"""
utexas/sat/tasks.py

Satisfiability task types.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import (
    abstractmethod,
    abstractproperty,
    )
from cargo.log   import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

class SAT_Task(ABC):
    """
    A satisfiability task.
    """

    @abstractproperty
    def name(self):
        """
        An arbitrary printable name for the task.
        """

class SAT_FileTask(SAT_Task):
    """
    A task backed by a .cnf file.
    """

    def __init__(self, path):
        """
        Initialize.
        """

        self.path = path

    @property
    def name(self):
        """
        An arbitrary printable name for the task.
        """

        return self.path

class SAT_MockTask(SAT_Task):
    """
    An abstract task.
    """

class SAT_MockFileTask(SAT_MockTask):
    """
    An abstract file task.
    """

    def __init__(self, task_row):
        """
        Initialize.
        """

        self.task_row = task_row

    @property
    def name(self):
        """
        An arbitrary printable name for the task.
        """

        return self.task_row.uuid

    def select_attempts(self):
        """
        Return a query selecting attempts on this task.
        """

        from sqlalchemy  import (
            and_,
            select,
            )
        from utexas.data import (
            SAT_TrialRow,
            SAT_AttemptRow,
            sat_attempts_trials_table as satt,
            )

        query = \
            select(
                SAT_AttemptRow.__table__.columns,
                and_(
                    SAT_AttemptRow.task_uuid == self.task_row.uuid,
                    SAT_AttemptRow.uuid      == satt.c.attempt_uuid,
                    satt.c.trial_uuid        == SAT_TrialRow.RECYCLABLE_UUID,
                    ),
                )

#         print "\n", select(query.alias().columns).compile()

        return query

class SAT_MockPreprocessedTask(SAT_MockTask):
    """
    An abstract preprocessed task.
    """

    def __init__(self, preprocessor_name, inner_task, from_):
        """
        Initialize.
        """

        self.preprocessor_name = preprocessor_name
        self.inner_task        = inner_task
        self.from_             = from_.alias()

    @property
    def name(self):
        """
        An arbitrary printable name for the task.
        """

        return "%s:%s" % (self.preprocessor_name, self.inner_task.name)

    def select_attempts(self):
        """
        Return a query selecting attempts on this task.
        """

        from sqlalchemy  import select
        from utexas.data import SAT_AttemptRow

        query = \
            select(
                SAT_AttemptRow.__table__.columns,
                SAT_AttemptRow.uuid == self.from_.c.inner_attempt_uuid,
                )

        print "\n", query

        return query

