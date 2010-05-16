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

# class SAT_Answer(object):
#     """
#     An answer (or lack thereof) to a SAT instance.
#     """

#     def __init__(self, satisfiable = None, certificate = None):
#         """
#         Initialize.
#         """

#         self.satisfiable = satisfiable
#         self.certificate = certificate

class SAT_Task(ABC):
    """
    A satisfiability task.
    """

    def to_orm(self, session):
        """
        Return a database description of this task.
        """

        raise RuntimeError("task has no database twin")

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

        SAT_Task.__init__(self)

        self._path = path

    @property
    def name(self):
        """
        An arbitrary printable name for the task.
        """

        return self._path

    @property
    def path(self):
        """
        The path of the associated task file.
        """

        return self._path

class SAT_PreprocessedTask(SAT_Task):
    """
    A task backed by a .cnf file and associated preprocessor information.
    """

    @abstractmethod
    def extend(self, certificate):
        """
        Extend the specified certificate.

        Translates a solution to the preprocessed CNF expression back into a
        solution to the unprocessed CNF expression.
        """

class SAT_MockTask(SAT_Task):
    """
    An non-real task with an associated database row.
    """

    def __init__(self, task_uuid):
        """
        Initialize.
        """

        SAT_Task.__init__(self)

        self._task_uuid = task_uuid

    def to_orm(self, session):
        """
        Return a database description of this task.
        """

        from utexas.data import TaskRow

        return session.query(TaskRow).get(self._task_uuid)

    @property
    def name(self):
        """
        An arbitrary printable name for the task.
        """

        return self._task_uuid

    @property
    def task_uuid(self):
        """
        The UUID of the associated database row.
        """

        return self._task_uuid

