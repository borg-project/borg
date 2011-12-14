"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import tempfile
import subprocess
import condor
import borg

logger = borg.get_logger(__name__, default_level = "DEBUG")

def ground_instance(asp_path, gringo_path, domain_path, ignore_errors):
    """Ground an ASP instance using Gringo."""

    (asp_path_base, _) = os.path.splitext(asp_path)
    verify_path = "{0}.verify".format(asp_path_base)
    command = [gringo_path, domain_path, asp_path]

    if os.path.exists(verify_path):
        command.append(verify_path)

        verified = " (and verified)"
    else:
        verified = ""

    logger.debug("running %s", command)

    with tempfile.TemporaryFile(suffix = ".gringo") as temporary_file:
        gringo_status = \
            subprocess.call(
                command,
                stdout = temporary_file,
                stderr = temporary_file,
                )

        temporary_file.flush()
        temporary_file.seek(0)

        if gringo_status != 0:
            message = "gringo failed to ground {0}".format(asp_path)

            if ignore_errors:
                logger.warning("%s", message)

                return None
            else:
                raise Exception(message)
        else:
            logger.info("grounded %s%s", asp_path, verified)

            return temporary_file.read()

@borg.annotations(
    gringo_path = ("path to Gringo", "positional", None, os.path.abspath),
    domain_path = ("path the domain file", "positional", None, os.path.abspath),
    root_path = ("instances root directory",),
    ignore_errors = ("ignore Gringo errors", "flag"),
    workers = ("number of Condor workers", "option", None, int),
    )
def main(gringo_path, domain_path, root_path, ignore_errors = False, workers = 0):
    """Ground a set of ASP instances using Gringo."""

    asp_paths = map(os.path.abspath, borg.util.files_under(root_path, "*.asp"))

    def yield_jobs():
        for asp_path in asp_paths:
            yield (ground_instance, [asp_path, gringo_path, domain_path, ignore_errors])

    for (job, lparse) in condor.do(yield_jobs(), workers):
        (asp_path, _, _, _) = job.args
        lparse_path = "{0}.lparse".format(asp_path)

        if lparse is not None:
            with open(lparse_path, "wb") as lparse_file:
                lparse_file.write(lparse)

if __name__ == "__main__":
    borg.script(main)

