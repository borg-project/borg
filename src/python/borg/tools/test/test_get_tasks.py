"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    assert_true,
    assert_equal,
    )

def assert_tasks_stored(session):
    """
    Verify that the tasks have been stored.
    """

    from borg.data import (
        TaskRow     as TR,
        TaskNameRow as TNR,
        )

    # we should have eight unique tasks
    assert_equal(session.query(TR).count(), 8)

    for i in xrange(8):
        for j in xrange(8):
            task_row =                                          \
                session                                         \
                .query(TNR)                                     \
                .filter(TNR.name == "tasks/%i/%i.cnf" % (i, j)) \
                .filter(TNR.collection == "sat/")               \
                .first()

            assert_true(task_row is not None)
            assert_true(task_row.task is not None)

def make_cnf(literals):
    """
    Make arbitrary (incorrect, but well-formed) CNF text.
    """

    return "p cnf 2 6\n%s 0\n" % " ".join(str(i) for i in literals)

def test_get_tasks():
    """
    Test the task acquisition tool.
    """

    # scan a fake path
    from cargo.io  import mkdtemp_scoped
    from cargo.log import get_logger

    log = get_logger(__name__, level = "NOTSET")

    with mkdtemp_scoped() as sandbox_path:
        # populate a directory with tasks
        from os      import makedirs
        from os.path import join

        tasks_path = join(sandbox_path, "tasks")
        task_paths = []

        for i in xrange(8):
            tasks_i_path = join(tasks_path, "%i" % i)

            makedirs(tasks_i_path)

            for j in xrange(8):
                task_ij_path = join(tasks_i_path, "%i.cnf" % j)

                with open(task_ij_path, "w") as file:
                    file.write(make_cnf([j]))

        # set up a test database
        from cargo.sql.alchemy import SQL_Engines

        #url = "sqlite:///%s" % join(sandbox_path, "test.sqlite")
        url = "sqlite:////tmp/baz.sqlite"

        with SQL_Engines() as engines:
            from borg.data import DatumBase

            DatumBase.metadata.create_all(engines.get(url).connect())

        # invoke the script
        from cargo.io import call_capturing
        from borg     import export_clean_defaults_path

        (stdout, stderr, code) = \
            call_capturing(
                [
                    "python",
                    "-m",
                    "borg.tools.get_tasks",
                    tasks_path,
                    sandbox_path,
					"--domain",
					"sat",
                    "--collection",
                    "sat/",
                    "-url",
                    url,
                    ],
                preexec_fn = export_clean_defaults_path,
                )

        log.debug("call stdout follows:\n%s", stdout)
        log.debug("call stderr follows:\n%s", stderr)

        assert_equal(code, 0)

        # success?
        with SQL_Engines() as engines:
            with engines.make_session(url)() as session:
                assert_tasks_stored(session)

