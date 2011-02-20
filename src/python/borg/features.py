"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re
import os.path
import cargo

logger = cargo.get_logger(__name__)

def get_features_for(cnf_path):
    """Obtain features of a CNF."""

    home = "/scratch/cluster/bsilvert/sat-competition-2011/solvers/features1s"
    command = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/run-1.4/run",
        "-k",
        os.path.join(home, "features1s"),
        cnf_path,
        ]

    def set_library_path():
        os.environ["LD_LIBRARY_PATH"] = "{0}:{1}".format(home, os.environ["LD_LIBRARY_PATH"])

    (stdout, stderr, code) = cargo.call_capturing(command, preexec_fn = set_library_path)

    match = re.search(r"^\[run\] time:[ \t]*(\d+.\d+) seconds$", stderr, re.M)
    (cost,) = map(float, match.groups())

    (names, values) = [l.split(",") for l in stdout.splitlines()[-2:]]
    values = map(float, values)

    logger.info("collected features for %s in %.2f s", cnf_path, cost)

    return (["cost"] + names, [cost] + values)

