"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_uncompressing_solver():
    """
    Test the uncompressing solver wrapper.
    """

    from tempfile                        import NamedTemporaryFile
    from utexas.sat.solvers.test.support import (
        TaskVerifyingSolver,
        unsanitized_cnf,
        )

    with NamedTemporaryFile(suffix = ".cnf.gz") as named_file:
        # write the compressed CNF expression
        from gzip       import GzipFile
        from contextlib import closing

        with closing(GzipFile(mode = "w", fileobj = named_file)) as gzip_file:
            gzip_file.write(unsanitized_cnf)

        named_file.flush()

        # test the solver
        from utexas.sat.tasks   import FileTask
        from utexas.sat.solvers import SAT_UncompressingSolver

        inner_solver = TaskVerifyingSolver(unsanitized_cnf)
        solver       = SAT_UncompressingSolver(inner_solver)
        task         = FileTask(named_file.name)

        solver.solve(task, None, None, None)

class TaskVerifyingPreprocessor(SAT_Preprocessor):
    """
    Preprocessor that merely verifies the contents of the task.
    """

    def __init__(self, correct_cnf):
        """
        Initialize.
        """

        self.correct_cnf = correct_cnf

    def preprocess(self, task, budget, output_dir, random, environment):
        """
        Verify behavior.
        """

        from utexas.sat.preprocessors import BarePreprocessorResult

        with open(task.path) as task_file:
            assert_equal(task_file.read(), self.correct_cnf)

        return BarePreprocessorResult(self, task, task, budget, budget, None)

    def extend(self, task, answer):
        """
        Pretend to extend an answer.
        """

        raise NotImplementedError()

def test_uncompressiong_preprocessor():
    """
    Test the uncompressing preprocessor wrapper.
    """

    # test
    from tempfile                        import NamedTemporaryFile
    from utexas.sat.solvers.test.support import sanitized_cnf

    with NamedTemporaryFile(suffix = ".cnf.gz") as named_file:
        # write the compressed CNF expression
        from gzip       import GzipFile
        from contextlib import closing

        with closing(GzipFile(mode = "w", fileobj = named_file)) as gzip_file:
            gzip_file.write(sanitized_cnf)

        named_file.flush()

        # test the solver
        from utexas.sat.tasks         import FileTask
        from utexas.sat.preprocessors import UncompressingPreprocessor

        task_verifying = TaskVerifyingPreprocessor(sanitized_cnf)
        uncompressing  = UncompressingPreprocessor(task_verifying)
        task           = FileTask(named_file.name)

        uncompressing.preprocess(task, None, None, None, None)

