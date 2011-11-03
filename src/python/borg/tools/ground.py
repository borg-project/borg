"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac
import borg

if __name__ == "__main__":
    plac.call(borg.tools.ground.main)

import os
import os.path
import subprocess
import cargo

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    gringo_path = ("path to Gringo",),
    domain_path = ("path the domain file",),
    root_path = ("instances root directory",),
    )
def main(gringo_path, domain_path, root_path):
    """Ground a set of ASP instances using Gringo."""

    cargo.enable_default_logging()

    # list relevant files
    asp_paths = map(os.path.abspath, cargo.files_under(root_path, "*.asp"))

    for asp_path in asp_paths:
        lparse_path = "{0}.lparse".format(asp_path)

        with open(lparse_path, "w") as lparse_file:
            gringo_status = \
                subprocess.call(
                    [gringo_path, domain_path, asp_path],
                    stdout = lparse_file,
                    stderr = lparse_file,
                    )

            if gringo_status != 0:
                raise Exception("gringo failed to ground \"{0}\"".format(asp_path))

        logger.info("grounded %s", asp_path)

