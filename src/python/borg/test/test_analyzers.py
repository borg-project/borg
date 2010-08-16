"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_no_analyzer():
    """
    Test the no-op analyzer.
    """

    from nose.tools     import assert_equal
    from borg.tasks     import FileTask
    from borg.solvers   import Environment
    from borg.analyzers import NoAnalyzer

    analyzer = NoAnalyzer()

    assert_equal(analyzer.feature_names, [])
    assert_equal(analyzer.analyze(FileTask("foo.cnf"), Environment()), {})

def get_file_task(relative):
    """
    Get a FileTask for some task stored in the support directory.
    """

    from borg       import get_support_path
    from borg.tasks import FileTask

    return FileTask(get_support_path("for_tests/s57-100.cnf"))

def assert_analyzer_ok(analyzer, task):
    """
    Assert than an analyzer behaves reasonably.
    """

    from nose.tools   import assert_true
    from borg.solvers import Environment

    features = analyzer.analyze(task, Environment())

    assert_true(len(analyzer.feature_names) > 0)

    for name in analyzer.feature_names:
        assert_true(features.has_key(name))

def test_satzilla_analyzer():
    """
    Test the SATzilla features1s wrapper analyzer.
    """

    from borg.analyzers import SATzillaAnalyzer

    assert_analyzer_ok(
        SATzillaAnalyzer(),
        get_file_task("for_tests/s57-100.cnf"),
        )

def test_uncompressing_analyzer():
    """
    Test the uncompressing analyzer wrapper.
    """

    from borg.analyzers import (
        SATzillaAnalyzer,
        UncompressingAnalyzer,
        )

    assert_analyzer_ok(
        UncompressingAnalyzer(SATzillaAnalyzer()),
        get_file_task("for_tests/s57-100.cnf.xz"),
        )

def assert_recycling_analyzer_ok(Session, session):
    """
    Assert that the recycling analyzer behaves correctly.
    """

    # insert some fake data
    from borg.data import (
        DatumBase,
        TaskRow        as TR,
        TaskFeatureRow as TFR,
        )

    DatumBase.metadata.create_all(session.connection())

    task_row = TR()

    session.add_all([
        TFR(task = task_row, name = "foo", value = False),
        TFR(task = task_row, name = "bar", value = True),
        ])
    session.commit()

    # verify the analyzer
    from nose.tools     import (
        assert_true,
        assert_equal,
        )
    from borg.tasks     import UUID_Task
    from borg.solvers   import Environment
    from borg.analyzers import RecyclingAnalyzer

    analyzer = RecyclingAnalyzer(["foo", "bar"])
    features = \
        analyzer.analyze(
            UUID_Task(task_row.uuid),
            Environment(CacheSession = Session),
            )

    assert_true(len(features) == 2)
    assert_equal(features["foo"], False)
    assert_equal(features["bar"], True)

def test_recycling_analyzer():
    """
    Test the recycling analyzer.
    """

    from cargo.sql.alchemy import (
        SQL_Engines,
        make_session,
        )

    with SQL_Engines() as engines:
        Session = make_session(bind = engines.get("sqlite:///:memory:"))

        with Session() as session:
            assert_recycling_analyzer_ok(Session, session)

