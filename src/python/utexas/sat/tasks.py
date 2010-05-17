"""
utexas/sat/tasks.py

Satisfiability task types.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc          import (
    abstractmethod,
    abstractproperty,
    )
from cargo.log    import get_logger
from utexas.rowed import (
    Rowed,
    AbstractRowed,
    )

log = get_logger(__name__)

class AbstractTask(AbstractRowed):
    """
    A task.
    """

    @abstractproperty
    def name(self):
        """
        An arbitrary printable name for the task.
        """

class AbstractFileTask(AbstractTask):
    """
    A task backed by a file.
    """

    @property
    def name(self):
        """
        A printable name for the task.
        """

        return self._path

    @abstractproperty
    def path(self):
        """
        The path to the associated task file.
        """

class AbstractPreprocessedTask(AbstractTask):
    """
    A preprocessed task.
    """

    @property
    def name(self):
        """
        A printable name for the task.
        """

        if self.seed is None:
            return "%s:%s" % (self.preprocessor, self.path)
        else:
            return "%s(%i):%s" % (self.preprocessor, self.seed, self.path)

    @abstractproperty
    def preprocessor(self):
        """
        The preprocessor that yielded this task.
        """

    @abstractproperty
    def seed(self):
        """
        The preprocessor seed on the run that yielded this task.
        """

    @abstractproperty
    def input_task(self):
        """
        The preprocessor input task that yielded this task.
        """

class AbstractPreprocessedFileTask(AbstractPreprocessedTask, AbstractFileTask):
    """
    A preprocessed task backed by a file.
    """

    @abstractproperty
    def output_path(self):
        """
        The path to the directory of preprocessor output files.
        """

class FileTask(AbstractFileTask, Rowed):
    """
    A task backed by a file.
    """

    def __init__(self, path, row = None):
        """
        Initialize.
        """

        Rowed.__init__(self, row)

        self._path = path

    @property
    def path(self):
        """
        The path of the associated task file.
        """

        return self._path

class MockTask(Rowed, AbstractTask):
    """
    A task not backed by a file.
    """

    def __init__(self, task_uuid):
        """
        Initialize.
        """

        self._task_uuid = task_uuid

    def get_row(self, session):
        """
        Get the ORM row associated with this object, if any.
        """

        from utexas.rowed import NoRowError

        try:
            return super(self).get_row(session)
        except NoRowError:
            from utexas.data import TaskRow

            row = session.query(TaskRow).get(self._task_uuid)

            if row is None:
                raise NoRowError()
            else:
                return row

    def add_row(self, session):
        """
        Create an ORM row for this object, if one does not already exist.
        """

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

