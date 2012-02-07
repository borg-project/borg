"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import resource
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
        else:
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

def get_features_for(asp_path, claspre_path):
    """Invoke claspre to compute features of an ASP instance."""

    previous_utime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime

    # get feature names
    (names_out, _) = borg.util.check_call_capturing([claspre_path, "--listFeatures", "-f", asp_path])
    (dynamic_names_out, static_names_out) = names_out.splitlines()

    dynamic_names = normalized_claspre_names(dynamic_names_out.split(","))
    static_names = normalized_claspre_names(static_names_out.split(","))

    # get features
    (values_out, _, _) = borg.util.call_capturing([claspre_path, "--claspfolio", "1", "-f", asp_path])
    values = [map(parse_claspre_value, line.split(",")) for line in values_out.splitlines()]

    # ...
    cost = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - previous_utime

    borg.get_accountant().charge_cpu(cost)

    logger.info("collected features of %s in %.2fs", asp_path, cost)

    assert len(static_names) == len(values[-1])

    return (static_names, values[-1])

