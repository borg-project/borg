"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import tempfile
import resource
import subprocess
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def normalized_claspre_names(raw_names):
    """Convert names from claspre to "absolute" names."""

    parent = None
    names = []

    for raw_name in raw_names:
        if raw_name.startswith("_"):
            assert parent is not None

            names.append(parent + raw_name)
        elif len(raw_name) > 0:
            names.append(raw_name)

            parent = raw_name

    return names

def parse_claspre_value(raw_value):
    """Convert values from claspre to floats."""

    special = {
        "No": -1.0,
        "Yes": 1.0,
        "NA": 0.0,
        }

    value = special.get(raw_value)

    if value is None:
        return float(raw_value)
    else:
        return value

def get_claspfolio_features_for(asp_path, binaries_path):
    """Invoke claspre to compute features of an ASP instance."""

    previous_utime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime

    # get feature names
    claspre_path = os.path.join(binaries_path, "claspfolio-0.8.0-x86-linux/clasp+pre-1.3.4")
    (names_out, _) = borg.util.check_call_capturing([claspre_path, "--list-features"])
    (dynamic_names_out, static_names_out) = names_out.splitlines()
    dynamic_names = normalized_claspre_names(dynamic_names_out.split(","))
    static_names = normalized_claspre_names(static_names_out.split(","))

    # compute feature values
    values_command = [
        claspre_path,
        "--rand-prob=10,30",
        "--search-limit=300,10",
        "--features=C1",
        "--file",
        asp_path,
        ]
    num_restarts = 10

    logger.info("running %s", values_command)

    (values_out, _, _) = borg.util.call_capturing(values_command)
    values_per = [map(parse_claspre_value, l.split(",")) for l in values_out.strip().splitlines()]

    if len(values_per) < num_restarts + 1:
        # claspre failed, or the instance was solved in preprocessing
        if len(values_per) == 0:
            # (claspre died)
            values_per = [[0.0] * len(static_names)]

        missing = (num_restarts - len(values_per) + 1)
        values_per = values_per[:-1] + ([[0.0] * len(dynamic_names)] * missing) + values_per[-1:]
    else:
        assert len(values_per) == num_restarts + 1

    # pull them together
    names = []
    values = []

    for i in xrange(num_restarts):
        names += ["restart{0}-{1}".format(i, n) for n in dynamic_names]
        values += values_per[i]

    names += static_names
    values += values_per[-1]

    # ...
    cost = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - previous_utime

    borg.get_accountant().charge_cpu(cost)

    logger.info("collected features of %s in %.2fs", asp_path, cost)

    assert len(names) == len(values)

    return (names, values)

def get_lp2sat_features_for(asp_path, binaries_path):
    """Convert to CNF and compute SAT features of an ASP instance."""

    with tempfile.NamedTemporaryFile(prefix = "borg.", suffix = ".cnf") as cnf_file:
        with open(asp_path, "rb") as asp_file:
            try:
                borg.domains.asp.run_lp2sat(binaries_path, asp_file, cnf_file)
            except borg.domains.asp.LP2SAT_FailedException:
                # XXX this workaround is silly; just improve sat.features
                cnf_file.seek(0)
                cnf_file.truncate(0)
                cnf_file.write("p cnf 1 1\n1 0\n")
                cnf_file.flush()

            return borg.domains.sat.features.get_features_for(cnf_file.name)

def get_features_for(asp_path, binaries_path):
    """Compute features of an ASP instance."""

    #(cnf_names, cnf_values) = get_lp2sat_features_for(asp_path, binaries_path)
    (clasp_names, clasp_values) = get_claspfolio_features_for(asp_path, binaries_path)

    #cnf_qnames = map("cnf-{0}".format, cnf_names)
    clasp_qnames = map("clasp-{0}".format, clasp_names)

    #return (cnf_qnames + clasp_qnames, cnf_values + clasp_values)
    return (clasp_qnames, clasp_values)

