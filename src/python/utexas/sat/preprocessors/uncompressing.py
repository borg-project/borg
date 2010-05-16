"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log                import get_logger
from utexas.sat.preprocessors import SAT_Preprocessor

log = get_logger(__name__)

class SAT_UncompressingPreprocessor(SAT_Preprocessor):
    """
    Uncompress and then preprocess SAT instances.
    """

    def __init__(self, preprocessor):
        """
        Initialize.
        """

        SAT_Preprocessor.__init__(self)

        self._inner = preprocessor

    def preprocess(self, task, budget, output_dir, random, environment):
        """
        Preprocess an instance.
        """

        # argument sanity
        from utexas.sat.tasks import SAT_FileTask

        if not isinstance(task, SAT_FileTask):
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
            inner_task = SAT_FileTask(uncompressed_path)

            return self._inner.preprocess(inner_task, budget, output_dir, random, environment)

