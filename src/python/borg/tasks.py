"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import (
    abstractmethod,
    abstractproperty,
    )
from contextlib  import contextmanager
from collections import namedtuple
from cargo.log   import get_logger
from borg.rowed  import (
    Rowed,
    AbstractRowed,
    )

log = get_logger(__name__)

DomainProperties = namedtuple("DomainProperties", ["patterns", "extension", "sanitizer"])

def get_builtin_domains():
    """
    Return the properties of built-in domains.
    """

    from borg.sat.cnf import yield_sanitized_cnf
    from borg.pb.opb  import yield_sanitized_opb

    return {
        "sat" : \
            DomainProperties(
                ["*.cnf", "*.cnf.gz", "*.cnf.bz2", "*.cnf.xz"],
                "cnf",
                yield_sanitized_cnf,
                ),
        "pb" : \
            DomainProperties(
                ["*.opb", "*.opb.gz", "*.opb.bz2", "*.opb.xz"],
                "opb",
                yield_sanitized_opb,
                ),
        }

builtin_domains = get_builtin_domains()

def get_task_file_hash(path, domain):
    """
    Return the hash of the specified task file.
    """

    from os.path  import join
    from cargo.io import (
        decompress_if,
        mkdtemp_scoped,
        hash_yielded_bytes,
        )

    with mkdtemp_scoped(prefix = "borg.tasks.") as sandbox_path:
        uncompressed_name = "uncompressed.%s" % domain.extension
        uncompressed_path = decompress_if(path, join(sandbox_path, uncompressed_name))

        with open(uncompressed_path) as file:
            (_, file_hash) = hash_yielded_bytes(domain.sanitizer(file), "sha512")

            return file_hash

def get_collections(path = defaults.collections, default = {None: "."}):
    """
    Get paths to task collections from a configuration file.
    """

    from cargo.io   import expandpath
    from cargo.json import load_json

    if path is None:
        return default
    else:
        return dict((k, expandpath(v)) for (k, v) in load_json(path))

@contextmanager
def uncompressed_task(task):
    """
    Provide an uncompressed task in a managed context.
    """

    # it it's not file-backed, pass it along
    from borg.tasks import AbstractFileTask

    if not isinstance(task, AbstractFileTask):
        yield task
    else:
        # create the context
        from cargo.io import mkdtemp_scoped

        with mkdtemp_scoped(prefix = "uncompressing.") as sandbox_path:
            # decompress the instance, if necessary
            from os.path  import join
            from cargo.io import decompress_if

            sandboxed_path    = join(sandbox_path, "uncompressed.cnf")
            uncompressed_path = decompress_if(task.path, sandboxed_path)

            log.info("maybe-decompressed %s to %s", task.path, uncompressed_path)

            # provide the task
            from borg.tasks import UncompressedFileTask

            yield UncompressedFileTask(uncompressed_path, task)

class AbstractTask(AbstractRowed):
    """
    Interface for a task.
    """

    @property
    def description(self):
        """
        Return an arbitrary description of this task.
        """

        return str(self)

#tasks = map(UUID_Task, TaskRow.with_prefix(session, "sat/competition_2009/random/"))

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

class UUID_Task(Task):
    """
    An unbacked task identified by UUID.
    """

    def __init__(self, uuid, row = None):
        """
        Initialize.
        """

        if row is not None and uuid != row.uuid:
            raise ValueError("uuid and row uuid do not match")

        Rowed.__init__(self, row)

        self._uuid = uuid

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        from borg.data import TaskRow as TR

        return session.query(TR).get(self._uuid)

    @staticmethod
    def with_prefix(session, prefix):
        """
        Return the task uuids associated with a name prefix.
        """

        from borg.data import TaskRow

        task_rows = TaskRow.with_prefix(session, prefix)

        return [UUID_Task(task_row.uuid) for task_row in task_rows]

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

class WrappedPreprocessedTask(WrappedTask, AbstractPreprocessedTask):
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

        WrappedTask.__init__(self, inner)

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
    def path(self):
        """
        The path to the associated task file.
        """

        return self._inner.path

    @property
    def output_path(self):
        """
        The path to the directory of preprocessor output files.
        """

        return self._inner.output_path

