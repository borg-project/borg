"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_uncompressing_solver():
    """
    Test the uncompressing solver wrapper.
    """

    from tempfile                  import NamedTemporaryFile
    from borg.solvers.test.support import (
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
        from borg.tasks   import FileTask
        from borg.solvers import UncompressingSolver

        inner_solver = TaskVerifyingSolver(unsanitized_cnf)
        solver       = UncompressingSolver(inner_solver)
        task         = FileTask(named_file.name)

        solver.solve(task, None, None, None)

def test_uncompressiong_preprocessor():
    """
    Test the uncompressing preprocessor wrapper.
    """

    # test
    from tempfile                  import NamedTemporaryFile
    from borg.solvers.test.support import (
        TaskVerifyingPreprocessor,
        sanitized_cnf,
        )

    with NamedTemporaryFile(suffix = ".cnf.gz") as named_file:
        # write the compressed CNF expression
        from gzip       import GzipFile
        from contextlib import closing

        with closing(GzipFile(mode = "w", fileobj = named_file)) as gzip_file:
            gzip_file.write(sanitized_cnf)

        named_file.flush()

        # test the solver
        from borg.tasks   import FileTask
        from borg.solvers import UncompressingPreprocessor

        task_verifying = TaskVerifyingPreprocessor(sanitized_cnf)
        uncompressing  = UncompressingPreprocessor(task_verifying)
        task           = FileTask(named_file.name)

        uncompressing.preprocess(task, None, None, None, None)

