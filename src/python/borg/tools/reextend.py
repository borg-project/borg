"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import os.path
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

@borg.annotations(
    root = ("root path for search", "positional"),
    old_suffix = ("old file suffix", "positional"),
    new_suffix = ("new file suffix", "positional"),
    pretend = ("new file suffix", "flag"),
    )
def main(root, old_suffix, new_suffix, pretend = False):
    """Change the extensions of multiple files."""

    for old_path in borg.util.files_under(root, [old_suffix]):
        new_path = old_path[:-len(old_suffix)] + new_suffix

        assert not os.path.exists(new_path)

        logger.info("renaming %s to %s", old_path, new_path)

        if not pretend:
            os.rename(old_path, new_path)

if __name__ == "__main__":
    borg.script(main)

