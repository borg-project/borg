"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def get_features_for(domain, task_path):
    """Read or compute features of a PB instance."""

    csv_path = task_path + ".features.csv"

    if os.path.exists(csv_path):
        features_array = numpy.recfromcsv(csv_path)
        features = features_array.tolist()

        assert features_array.dtype.names[0] == "cpu_cost"

        borg.get_accountant().charge_cpu(features[0])
    else:
        (_, features) = domain.compute_features(task_path)

    return features

