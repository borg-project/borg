"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.get_targets import main

    plac.call(main)

import os.path
import csv
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    out_path = ("results output path"),
    suite_path = ("path to the solvers suite", "positional", None, os.path.abspath),
    instances_root = ("path to instances", "positional", None, os.path.abspath),
    suffix = ("runs data file suffix", "option"),
    )
def main(
    out_path,
    suite_path,
    instances_root,
    suffix = ".runs.csv",
    ):
    """Evaluate portfolio performance under a specified model."""

    cargo.enable_default_logging()

    suite = borg.load_solvers(suite_path)
    paths = list(cargo.files_under(instances_root, suite.domain.extensions))

    logger.info("reading from %i instance(s) under %s", len(paths), instances_root)

    run_data = borg.RunData.from_paths(paths, suite.domain, suffix)

    logger.info("building model given %i training runs", run_data.get_run_count())

    B = 10
    solver_names = list(suite.solvers)
    model = borg.models.Mul_ModelFactory().fit(solver_names, run_data, B)

    logger.info("computing and writing out responsibilities")

    counts = run_data.to_bins_array(solver_names, B)
    log_ps_NM = borg.models.sampled_pmfs_log_pmf(model.log_masses, counts)

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        #writer.writerow(["instance", "feature_name", "feature_value", "distribution", "log_probability"])
        writer.writerow(["instance", "distribution", "responsibility"])

        run_data_items = run_data.run_lists.items()

        for n in xrange(log_ps_NM.shape[0]):
            (instance_id, _) = run_data_items[n]

            for m in xrange(log_ps_NM.shape[1]):
                writer.writerow([instance_id, m, log_ps_NM[n, m]])

            #for (feature_name, feature_value) in run_data.get_feature_vector(instance_id).items()[:1]:
                #for m in xrange(log_ps_NM.shape[1]):
                    #writer.writerow([instance_id, feature_name, feature_value, m, log_ps_NM[n, m]])

