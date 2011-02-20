"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_copy_tables():
    """
    Test copying of arbitrary database tables.
    """

    from sqlalchemy        import create_engine
    from cargo.sql.alchemy import make_session
    from borg.data       import (
        DatumBase,
        AttemptRow,
        RunAttemptRow,
        CPU_LimitedRunRow,
        )
    from borg.cache      import copy_tables

    # set up the two databases
    from_engine = create_engine("sqlite:///:memory:")
    to_engine   = create_engine("sqlite:///:memory:")

    FromSession = make_session(bind = from_engine)
    ToSession   = make_session(bind = to_engine)

    from_session = FromSession()
    to_session   = ToSession()

    DatumBase.metadata.create_all(from_engine)
    DatumBase.metadata.create_all(to_engine)

    # insert some fake data
    fake_attempts = [
        RunAttemptRow(run = CPU_LimitedRunRow(), solver_name = "foo"),
        RunAttemptRow(run = CPU_LimitedRunRow(), solver_name = "foo"),
        RunAttemptRow(run = CPU_LimitedRunRow(), solver_name = "foo"),
        RunAttemptRow(run = CPU_LimitedRunRow(), solver_name = "foo"),
        ]

    from_session.add_all(fake_attempts)
    from_session.commit()

    # and test the copy operation
    from_connection = from_engine.contextual_connect()
    to_connection   = to_engine.contextual_connect()

    with to_connection.begin():
        copy_tables(from_connection, to_connection, DatumBase.metadata.sorted_tables)

    # did it work?
    assert_equal(to_session.query(AttemptRow).count(),        len(fake_attempts))
    assert_equal(to_session.query(RunAttemptRow).count(),     len(fake_attempts))
    assert_equal(to_session.query(CPU_LimitedRunRow).count(), len(fake_attempts))

