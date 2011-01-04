"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac
import borg

if __name__ == "__main__":
    from borg.tools.sanitize_cnfs import main

    plac.call(main)

import os
import shutil
import tempfile
import cargo.log

logger = cargo.log.get_logger(__name__, level = "NOTSET")

@plac.annotations(
    paths = ("CNFs to sanitize"),
    )
def main(*paths):
    """
    Run the script.
    """

    # set up logging
    cargo.log.enable_default_logging()

    # sanitize the CNFs
    for path in paths:
        with tempfile.NamedTemporaryFile(delete = False) as temporary:
            with open(path) as opened:
                borg.write_sanitized_cnf(opened, temporary)

            temporary.flush()

            os.fsync(temporary)

            shutil.move(temporary.name, path)

        logger.info("sanitized %s", path)

