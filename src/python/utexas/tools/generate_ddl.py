"""
utexas/tools/generate_ddl.py

Print or apply the research data schema.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.generate_ddl import main

    raise SystemExit(main())

from cargo.flags import (
    Flag,
    Flags,
    )

module_flags = \
    Flags(
        "Research Data Storage",
        Flag(
            "-a",
            "--apply",
            action = "store_true",
            help   = "create the generated schema",
            ),
        Flag(
            "-r",
            "--reflect",
            action = "store_true",
            help   = "load the reflected schema",
            ),
        Flag(
            "-t",
            "--topological",
            action = "store_true",
            help   = "print topologically sorted by dependency",
            ),
        )

def main():
    """
    Create core database metadata.
    """

    # connect to the database
    from cargo.flags import parse_given
    from utexas.data import research_connect

    parse_given(usage = "usage: %prog [options]")

    engine = research_connect()

    # load the appropriate schema
    if module_flags.given.reflect:
        # use the database's schema
        from sqlalchemy.schema import MetaData

        metadata = MetaData()

        metadata.reflect(bind = engine)
    else:
        # use the project-defined schema
        from utexas.data import DatumBase

        metadata = DatumBase.metadata

    # then do something with it
    if module_flags.given.apply:
        # apply the DDL to the database
        metadata.create_all(engine)
    else:
        # print the DDL
        from sqlalchemy.schema import CreateTable

        if module_flags.given.topological:
            sorted_tables = metadata.sorted_tables
        else:
            sorted_tables = sorted(metadata.sorted_tables, key = lambda t: t.name)

        for table in sorted_tables:
            print CreateTable(table).compile(engine)

