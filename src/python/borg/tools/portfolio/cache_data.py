"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.portfolio.cache_data import main

    raise SystemExit(main())

import borg.data

from cargo.flags    import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.log      import get_logger

log          = get_logger(__name__)
module_flags = \
    Flags(
        "Cache Construction Options",
        Flag(
            "-v",
            "--verbose",
            action  = "store_true",
            help    = "be noisier [%default]",
            ),
        )

@with_flags_parsed(
    usage = "usage: %prog [options] <cache>",
    )
def main((cache_path,)):
    """
    Main.
    """

    # basic flag handling
    from cargo.log import enable_default_logging
    from borg.data import (
        DatumBase,
        research_connect,
        )

    flags = module_flags.given

    if flags.verbose:
        get_logger("borg.cache", level = "DEBUG")

    enable_default_logging()

    # write the cache
    from sqlalchemy import create_engine
    from borg.cache import copy_tables

    from_engine     = research_connect()
    from_connection = from_engine.contextual_connect()

    to_engine     = create_engine("sqlite:///%s" % cache_path)
    to_connection = to_engine.contextual_connect()

    DatumBase.metadata.create_all(to_connection)

    with to_connection.begin():
        copy_tables(from_connection, to_connection, DatumBase.metadata.sorted_tables)

