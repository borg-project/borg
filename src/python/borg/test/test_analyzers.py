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

    assert_equal(analyzer.features, [])
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

    from nose.tools   import (
        assert_true,
        assert_equal,
        )
    from borg.solvers import Environment

    analysis = analyzer.analyze(task, Environment())

    assert_true(len(analyzer.features) > 0)

    for feature in analyzer.features:
        assert_true(analysis.has_key(feature.name))

    assert_equal(len(analyzer.features), len(analysis))

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
        TaskRow             as TR,
        TaskFloatFeatureRow as TFFR,
        )

    DatumBase.metadata.create_all(session.connection())

    task_row = TR()

    session.add_all([
        TFFR(task = task_row, name = "foo", value =  0.1),
        TFFR(task = task_row, name = "bar", value = -1e2),
        ])
    session.commit()

    # verify the analyzer
    from nose.tools     import (
        assert_true,
        assert_equal,
        )
    from borg.tasks     import UUID_Task
    from borg.solvers   import Environment
    from borg.analyzers import (
        Feature,
        RecyclingAnalyzer,
        )

    analyzer = RecyclingAnalyzer([Feature("foo", float), Feature("bar", float)])
    features = \
        analyzer.analyze(
            UUID_Task(task_row.uuid),
            Environment(CacheSession = Session),
            )

    assert_true(len(features) == 2)
    assert_equal(features["foo"],  0.1)
    assert_equal(features["bar"], -1e2)

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

