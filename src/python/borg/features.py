"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re
import os.path
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def get_features_for_cnf(cnf_path):
    """Obtain features of a SAT instance."""

    command = [
        os.path.join(borg.defaults.solvers_root, "run-1.4/run"),
        "-k",
        os.path.join(borg.defaults.solvers_root, "features1s/features1s"),
        cnf_path,
        ]

    def set_library_path():
        ld_library_path = os.path.join(borg.defaults.solvers_root, "features1s")

        if "LD_LIBRARY_PATH" in os.environ:
            ld_library_path += ":{0}".format(os.environ["LD_LIBRARY_PATH"])

        os.environ["LD_LIBRARY_PATH"] = ld_library_path

    (stdout, stderr, code) = cargo.call_capturing(command, preexec_fn = set_library_path)

    match = re.search(r"^\[run\] time:[ \t]*(\d+.\d+) seconds$", stderr, re.M)
    (cost,) = map(float, match.groups())

    (names, values) = [l.split(",") for l in stdout.splitlines()[-2:]]
    values = map(float, values)

    logger.info("collected features for %s in %.2f s", os.path.basename(cnf_path), cost)

    return (["cost"] + names, [cost] + values)

def get_features_for_opb(opb_path):
    """Obtain features of a PB instance."""

    # parse the instance and compute features
    with accounting() as accountant:
        with open(opb_path) as opb_file:
            opb = borg.opb.parse_opb_file(opb_file)

        features = {
            "optimization" : 0.0 if opb.objective is None else 1.0,
            "constraints" : len(opb.constraints),
            }

    # account for and return them
    cost = accountant.cpu_seconds

    logger.info("computed features for %s in %.2f s", os.path.basename(opb_path), cost)

    return (["cost"] + features.keys(), [cost] + features.values())

