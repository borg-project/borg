"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import re
import os.path
import numpy
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

def get_features_for(task_path):
    """Read or compute features of a PB instance."""

    csv_path = task_path + ".features.csv"

    if os.path.exists(csv_path):
        features_array = numpy.recfromcsv(csv_path)
        features = features_array.tolist()

        assert features_array.dtype.names[0] == "cpu_cost"

        borg.get_accountant().charge_cpu(features[0])
    else:
        (_, features) = borg.features.pb.path_compute_all(task_path)

    return features

