"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                    import call
    from borg.tools.generate_ddl import main

    call(main)

from plac import annotations

def generate_ddl(engine, reflect, apply, topological):
    """
    Print or apply the database schema.
    """

    # load the appropriate schema
    if reflect:
        # use the database's schema
        from sqlalchemy.schema import MetaData

        metadata = MetaData(bind = engine, reflect = True)
    else:
        # use the project-defined schema
        from borg.data import DatumBase

        metadata = DatumBase.metadata

    # then do something with it
    if apply:
        # apply the DDL to the database
        metadata.create_all(engine)
    else:
        # print the DDL
        from sqlalchemy.schema import CreateTable

        if topological:
            sorted_tables = metadata.sorted_tables
        else:
            sorted_tables = sorted(metadata.sorted_tables, key = lambda t: t.name)

        for table in sorted_tables:
            print CreateTable(table).compile(engine)

@annotations(
    reflect     = ("load the reflected schema"  , "flag", "r"),
    apply       = ("create the generated schema", "flag", "a"),
    topological = ("sort by dependency"         , "flag", "t"),
    )
def main(reflect = False, apply = False, topological = False):
    """
    Deal with core database metadata.
    """

    # be verbose in non-print modes
    if apply:
        import logging

        from cargo.log import (
            get_logger,
            enable_default_logging,
            )

        enable_default_logging()

        get_logger("sqlalchemy.engine").setLevel(logging.DEBUG)

    # connect to the database and go
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        from borg.data import research_connect

        generate_ddl(research_connect())

