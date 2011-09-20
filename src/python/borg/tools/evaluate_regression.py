"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.evaluate_regression import main

    plac.call(main)

import os.path
import csv
import numpy
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
    solver_names = list(suite.solvers)

    logger.info("reading from %i instance(s) under %s", len(paths), instances_root)

    run_data = borg.RunData.from_paths(solver_names, paths, suite.domain, suffix)

    logger.info("building model given %i training runs", run_data.get_run_count())

    B = 10
    model = borg.models.Mul_ModelFactory().fit(solver_names, run_data, B)

    logger.info("preparing regression")

    (feature_values_NF, ps_NM) = borg.regression.prepare_training_data_raw(run_data, model)
    #regress = borg.regression.LinearRegression(run_data, model)
    #regress = borg.regression.ConstantRegression(run_data, model)
    regress = borg.regression.ClusterRegression(run_data, model)

    logger.info("writing examples for RTD prediction")

    print run_data.common_features
    f = run_data.common_features.index("nvars")

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["index", "actual", "predicted", "nvars"])
        #writer.writerow(["probability", "nvars"])

        #for c in xrange(feature_values_NF.shape[0]):
        for c in [3]:
            for n in xrange(feature_values_NF.shape[0]):
                print c, n
                prediction = regress.predict(None, feature_values_NF[n])

                writer.writerow([c, ps_NM[n, c], prediction[c], feature_values_NF[n, f]])
                #writer.writerow([c, ps_NM[n, c], prediction[c]])
                #writer.writerow([ps_NM[n, c], feature_values_NF[n, f]])

