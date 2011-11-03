"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import resource
import cargo

logger = cargo.get_logger(__name__, default_level = "INFO")

def get_features_for(asp_path):
    """Obtain features of a CNF."""

    previous_utime = resource.getrusage(resource.RUSAGE_SELF).ru_utime

    # XXX
    #with open(asp_path) as cnf_file:
        #instance = borg.domains.sat.instance.parse_sat_file(cnf_file)

    cost = resource.getrusage(resource.RUSAGE_SELF).ru_utime - previous_utime

    #logger.info("parsed %s in %.2f s", cnf_path, cost)

    # XXX

    cost = resource.getrusage(resource.RUSAGE_SELF).ru_utime - previous_utime

    logger.info("collected features for %s in %.2f s", asp_path, cost)

    # XXX
    #return zip(*core_features)
    return ([], [])

