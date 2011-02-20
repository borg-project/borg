"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log import get_logger

log = get_logger(__name__)

def copy_table(from_connection, to_connection, table):
    """
    Copy a table from one engine to another.
    """

    from itertools  import izip
    from sqlalchemy import (
        select,
        insert,
        )

    # FIXME make the exclusion mechanism generic

    if table.name == "cpu_limited_runs":
        columns = [c for c in table.columns if c.name != "stderr" and c.name != "stdout"]
    elif table.name == "sat_answers":
        columns = [c for c in table.columns if c.name != "certificate_xz"]
    else:
        columns = table.columns

    column_names = [c.name for c in columns]

    log.detail("from %s, retrieving %s", table.name, column_names)

    result    = from_connection.execute(select(columns))
    statement = table.insert(dict((c, None) for c in columns))

    log.detail("inserting via statement: %s", statement)

    while True:
        rows = result.fetchmany(8192)

        if rows:
            log.detail("inserting %i row(s) into %s", len(rows), table)

            to_connection.execute(statement, rows)
        else:
            break

def copy_tables(from_connection, to_connection, tables):
    """
    Copy a set of tables from one session to another.
    """

    # copy over data
    for table in tables:
        copy_table(from_connection, to_connection, table)

