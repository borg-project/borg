"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import numpy
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

@borg.annotations(
    out_root = ("results output path"),
    bundle = ("path to pre-recorded runs", "positional", None, os.path.abspath),
    )
def main(out_root, bundle):
    """Write the latent classes of a matrix model."""

    # fit the model
    run_data = borg.storage.RunData.from_bundle(bundle)

    logger.info("fitting matrix mixture model")

    # extract the latent classes
    estimator = borg.models.MulDirMatMixEstimator(K = 16)
    model = estimator(run_data, 10, run_data)
    latent = model.latent_classes
    latent /= numpy.sum(latent, axis = -1)[..., None]

    (K, S, D) = latent.shape

    latent_csv_path = os.path.join(out_root, "latent_classes.csv.gz")

    with open(latent_csv_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["k", "solver", "bin", "value"])

        for k in xrange(K):
            for s in xrange(S):
                solver_name = run_data.solver_names[s]

                for d in xrange(D):
                    writer.writerow([k, solver_name, d, latent[k, s, d]])

    # extract the responsibilities
    (K, N) = model.responsibilities.shape

    responsibilities = numpy.exp(model.responsibilities)

    responsibilities_csv_path = os.path.join(out_root, "responsibilities.csv.gz")

    with open(responsibilities_csv_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["k", "n", "value"])

        for k in xrange(K):
            for n in xrange(N):
                writer.writerow([k, n, responsibilities[k, n]])

    # extract the categories
    responsibilities_csv_path = os.path.join(out_root, "categories.csv.gz")

    with open(responsibilities_csv_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["n", "category"])

        for (n, name) in enumerate(sorted(run_data.ids)):
            category = name.split("/")[8]

            writer.writerow([n, category])

if __name__ == "__main__":
    borg.script(main)

