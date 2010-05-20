"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc        import (
    abstractmethod,
    abstractproperty,
    )
from cargo.log  import get_logger
from borg.rowed import (
    Rowed,
    AbstractRowed,
    )

log = get_logger(__name__)

class AbstractTask(AbstractRowed):
    """
    Interface for a task.
    """

class AbstractFileTask(AbstractTask):
    """
    Interface for a task backed by a file.
    """

    @abstractproperty
    def path(self):
        """
        The path to the associated task file.
        """

class AbstractPreprocessedTask(AbstractTask):
    """
    Interface for a preprocessed task.
    """

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

class Task(Rowed, AbstractTask):
    """
    A task not necessarily backed by a file.
    """

    def __init__(self, row = None):
        """
        Initialize.
        """

        Rowed.__init__(self, row)

class WrappedTask(Rowed, AbstractTask):
    """
    A wrapped task.
    """

    def __init__(self, inner, row = None):
        """
        Initialize.
        """

        assert isinstance(inner, AbstractTask)

        Rowed.__init__(self, row)

        self._inner = inner

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        return self._inner.get_row(session)

class FileTask(Task, AbstractFileTask):
    """
    A task backed by a file.
    """

    def __init__(self, path, row = None):
        """
        Initialize.
        """

        Task.__init__(self, row)

        self._path = path

    @property
    def path(self):
        """
        The path to the associated task file.
        """

        return self._path

class UncompressedFileTask(AbstractFileTask):
    """
    An uncompressed view of a file-backed task.
    """

    def __init__(self, path, compressed):
        """
        Initialize.
        """

        assert isinstance(compressed, AbstractFileTask)

        self._path       = path
        self._compressed = compressed

    @property
    def path(self):
        """
        The path to the associated task file.
        """

        return self._path

    def get_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        return self._compressed.get_row(session)

    def set_row(self, row):
        """
        Set the ORM row associated with this object.
        """

        return self._compressed.set_row(row)

class WrappedFileTask(WrappedTask, AbstractFileTask):
    """
    A wrapped task.
    """

    @property
    def path(self):
        """
        The path to the associated task file.
        """

        return self._inner.path

class PreprocessedTask(Task, AbstractPreprocessedTask):
    """
    A preprocessed task backed by a directory.
    """

    def __init__(self, preprocessor, seed, input_task, row = None):
        """
        Initialize.
        """

        Task.__init__(self, row = row)

        self._preprocessor = preprocessor
        self._seed         = seed
        self._input_task   = input_task

    def get_new_row(self, session, preprocessor_row = None):
        """
        Create or obtain an ORM row for this object.
        """

        from sqlalchemy import and_
        from borg.data  import PreprocessedTaskRow as PT

        if preprocessor_row is None:
            preprocessor_row = self._preprocessor.get_row(session)

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

class WrappedPreprocessedTask(WrappedFileTask, AbstractPreprocessedTask, AbstractFileTask):
    """
    A wrapped task.
    """

    def __init__(self, preprocessor, inner):
        """
        Initialize.
        """

        from borg.solvers import AbstractPreprocessor

        assert isinstance(preprocessor, AbstractPreprocessor)
        assert isinstance(inner, AbstractPreprocessedTask)

        WrappedFileTask.__init__(self, inner)

        self._preprocessor = preprocessor

    def get_new_row(self, session, preprocessor_row = None):
        """
        Create or obtain an ORM row for this object.
        """

        if preprocessor_row is None:
            preprocessor_row = self._preprocessor.get_row(session)

        return self._inner.get_row(session, preprocessor_row = preprocessor_row)

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

        return self._inner.seed

    @property
    def input_task(self):
        """
        The preprocessor input task that yielded this task.
        """

        return self._inner.input_task

class PreprocessedDirectoryTask(PreprocessedTask, AbstractPreprocessedDirectoryTask):
    """
    A preprocessed task backed by a directory.
    """

    def __init__(self, preprocessor, seed, input_task, output_path, relative_task_path, row = None):
        """
        Initialize.
        """

        PreprocessedTask.__init__(self, preprocessor, seed, input_task, row = row)

        self._output_path        = output_path
        self._relative_task_path = relative_task_path

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

class WrappedPreprocessedDirectoryTask(WrappedPreprocessedTask, AbstractPreprocessedDirectoryTask):
    """
    A wrapped preprocessed task backed by a directory.
    """

    def __init__(self, preprocessor, inner):
        """
        Initialize.
        """

        assert isinstance(inner, AbstractPreprocessedDirectoryTask)

        WrappedPreprocessedTask.__init__(self, preprocessor, inner)

    @property
    def output_path(self):
        """
        The path to the directory of preprocessor output files.
        """

        return self._inner.output_path

