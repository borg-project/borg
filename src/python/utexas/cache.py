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

    result       = from_connection.execute(select(table.columns))
    column_names = [c.name for c in table.columns]

    while True:
        rows = result.fetchmany(8192)

        if rows:
            log.detail("inserting %i rows into %s", len(rows), table)

            to_connection.execute(table.insert(column_names), rows)
        else:
            break

def copy_tables(from_connection, to_connection, tables):
    """
    Copy a set of tables from one session to another.
    """

    # copy over data
    for table in tables:
        copy_table(from_connection, to_connection, table)

