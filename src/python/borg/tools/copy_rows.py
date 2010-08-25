"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                 import call
    from borg.tools.copy_rows import main

    call(main)

from plac      import annotations
from cargo.log import get_logger

log = get_logger(__name__, level = "NOTSET")

@annotations(
    quiet     = ("be less noisy"   , "flag"  , "q"),
    to_url    = ("destination URL"),
    from_urls = ("source URL(s)")  ,
    tables    = ("only table(s)"   , "option", "t" , lambda s: s.split(", ")),
    where     = ("SQL filter"      , "option", "w"),
    fetch     = ("buffer size"     , "option", "f" , int),
    )
def main(to_url, tables = None, quiet = False, where = None, fetch = 8192, *from_urls):
    """
    Copy data from source database(s) to some single target.
    """

    # basic setup
    from cargo.log import enable_console_log

    enable_console_log()

    if not quiet:
        get_logger("sqlalchemy.engine", level = "WARNING")
        get_logger("cargo.sql.actions", level = "DETAIL")

    if where is not None and (tables is None or len(tables) != 1):
        raise ValueError("exactly one table must be specified with where clause")

    # copy as requested
    from cargo.sql.alchemy import make_engine

    to_engine     = make_engine(to_url)
    to_connection = to_engine.connect()

    with to_connection.begin():
        if tables is not None:
            log.debug("permitting only tables: %s", tables)

        for from_url in from_urls:
            # normalize the URL
            log.info("copying from %s, fetching %i at a time", from_url, fetch)

            # connect to this source
            from_engine     = make_engine(from_url)
            from_connection = from_engine.connect()

            # reflect its schema
            from borg.data import DatumBase

            # copy its data
            for sorted_table in DatumBase.metadata.sorted_tables:
                if tables is None or sorted_table.name in tables:
                    log.debug("copying table %s", sorted_table.name)

                    from cargo.sql.actions import copy_table

                    copy_table(
                        from_connection,
                        to_connection,
                        sorted_table,
                        where = where,
                        fetch = fetch,
                        )

            # done
            from_engine.dispose()

    # done
    to_engine.dispose()

