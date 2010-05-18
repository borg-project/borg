"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log                import get_logger
from utexas.sat.tasks         import AbstractFileTask
from utexas.sat.preprocessors import SAT_Preprocessor
from utexas.rowed             import Rowed

log = get_logger(__name__)

class UncompressingPreprocessor(Rowed, SAT_Preprocessor):
    """
    Uncompress and then preprocess SAT instances.
    """

    def __init__(self, preprocessor):
        """
        Initialize.
        """

        Rowed.__init__(self)

        self._inner = preprocessor

    def preprocess(self, task, budget, output_dir, random, environment):
        """
        Preprocess an instance.
        """

        # argument sanity
        from utexas.sat.tasks import AbstractFileTask

        if not isinstance(task, AbstractFileTask):
            raise TypeError("uncompressing preprocessor requires a file-backed task")

        # preprocess
        from cargo.io import mkdtemp_scoped

        log.info("starting to preprocess %s", task.path)

        with mkdtemp_scoped(prefix = "uncompressing.") as sandbox_path:
            # decompress the instance, if necessary
            from os.path  import join
            from cargo.io import decompress_if

            uncompressed_path = \
                decompress_if(
                    task.path,
                    join(sandbox_path, "uncompressed.cnf"),
                    )

            log.info("uncompressed task is %s", uncompressed_path)

            # then pass it along
            inner_task = UncompressedFileTask(uncompressed_path, task)

            return self._inner.preprocess(inner_task, budget, output_dir, random, environment)

    def extend(self, task, answer):
        """
        Pretend to extend an answer.
        """

        return self._inner.extend(task, answer)

    def get_new_row(self, session, **kwargs):
        """
        Create or obtain an ORM row for this object.
        """

        return self._inner.get_new_row(session, **kwargs)

class UncompressedFileTask(AbstractFileTask):
    """
    An uncompressed view of a task backed by a file.
    """

    def __init__(self, uncompressed_path, compressed_task):
        """
        Initialize.
        """

        self._uncompressed_path = uncompressed_path
        self._compressed_task   = compressed_task

    @property
    def path(self):
        """
        The path to the associated uncompressed task file.
        """

        return self._uncompressed_path

    def get_row(self, session, **kwargs):
        """
        Get the ORM row associated with this object, if any.
        """

        return self._compressed_task.get_row(session, **kwargs)

    def set_row(self, row):
        """
        Set the ORM row associated with this object.
        """

        self._compressed_task.set_row(row)

