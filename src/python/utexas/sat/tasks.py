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
    Interface for a task.
    """

    @abstractproperty
    def name(self):
        """
        An arbitrary printable name for the task.
        """

class MockTask(Rowed, AbstractTask):
    """
    A task not backed by a file.
    """

    def __init__(self, task_uuid):
        """
        Initialize.
        """

        self._task_uuid = task_uuid

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

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

class AbstractFileTask(AbstractTask):
    """
    Interface for a task backed by a file.
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

class FileTask(Rowed, AbstractFileTask):
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
        The path to the associated task file.
        """

        return self._path

class AbstractPreprocessedTask(AbstractTask):
    """
    Interface for a preprocessed task.
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

class AbstractPreprocessedDirectoryTask(AbstractPreprocessedTask, AbstractFileTask):
    """
    Interface for a preprocessed task backed by a directory.
    """

    @abstractproperty
    def output_path(self):
        """
        The path to the directory of preprocessor output files.
        """

class PreprocessedDirectoryTask(Rowed, AbstractPreprocessedDirectoryTask):
    """
    A preprocessed task backed by a directory.
    """

    def __init__(self, preprocessor, seed, input_task, output_path, relative_task_path, row = None):
        """
        Initialize.
        """

        Rowed.__init__(self, row)

        self._preprocessor       = preprocessor
        self._seed               = seed
        self._input_task         = input_task
        self._output_path        = output_path
        self._relative_task_path = relative_task_path

    def get_new_row(self, session, preprocessor_row = None, **kwargs):
        """
        Get or create the ORM row associated with this object.
        """

        from sqlalchemy  import and_
        from utexas.data import PreprocessedTaskRow as PT

        if preprocessor_row is None:
            preprocessor_row = self.preprocessor.get_row(session)

        input_task_row         = self.input_task.get_row(session)
        preprocessed_task_row  =                         \
            session                                      \
            .query(PT)                                   \
            .filter(
                and_(
                    PT.preprocessor == preprocessor_row,
                    PT.seed         == self.seed,
                    PT.input_task   == input_task_row,
                ),
            )                                            \
            .first()

        if preprocessed_task_row is None:
            preprocessed_task_row = \
                PT(
                    preprocessor = preprocessor_row,
                    seed         = self.seed,
                    input_task   = input_task_row,
                    )

            session.add(preprocessed_task_row)

        return preprocessed_task_row

    @property
    def preprocessor(self):
        """
        The preprocessor that yielded this task.
        """

        return self._preprocessor

    @property
    def seed(self):
        """
        The preprocessor seed on the run that yielded this task.
        """

        return self._seed

    @property
    def input_task(self):
        """
        The preprocessor input task that yielded this task.
        """

        return self._input_task

    @property
    def path(self):
        """
        The path to the associated task file.
        """

        from os.path import join

        return join(self._output_path, self._relative_task_path)

    @property
    def output_path(self):
        """
        The path to the directory of preprocessor output files.
        """

        return self._output_path

