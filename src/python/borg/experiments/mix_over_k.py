"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import uuid
import numpy
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def evaluate_split(run_data, model_name, K, split, train_mask, test_mask):
    """Evaluate a model on a train/test split."""

    training = run_data.masked(train_mask).collect_systematic([4])
    testing = run_data.masked(test_mask).collect_systematic([4])

    # build the model
    if model_name == "mul-dirmix":
        estimator = borg.models.MulDirMixEstimator(K = K, samples_per = 128)
    elif model_name == "mul-dirmatmix":
        estimator = borg.models.MulDirMatMixEstimator(K = K)
    else:
        raise ValueError("unrecognized model name {0}".format(model_name))

    bins = 10
    model = estimator(training, bins, training)

    # evaluate the model
    score = numpy.mean(borg.models.run_data_log_probabilities(model, testing))

    logger.info(
        "%s score at K = %i given %i runs from %i instances: %f",
        model_name,
        K,
        training.get_run_count(),
        len(training),
        score,
        )

    return [model_name, K, len(training), split, score]

@borg.annotations(
    out_path = ("results output path"),
    bundle = ("path to pre-recorded runs", "positional", None, os.path.abspath),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, bundle, workers = 0, local = False):
    """Evaluate the mixture model(s) over a range of component counts."""

    def yield_jobs():
        run_data = borg.storage.RunData.from_bundle(bundle)
        validation = sklearn.cross_validation.ShuffleSplit(len(run_data), 64, test_fraction = 0.2, indices = False)

        for (train_mask, test_mask) in validation:
            split = uuid.uuid4()
            Ks = range(1, 64, 1)

            for K in Ks:
                for model_name in ["mul-dirmix", "mul-dirmatmix"]:
                    yield (evaluate_split, [run_data, model_name, K, split, train_mask, test_mask])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["model_name", "components", "instances", "split", "mean_log_probability"])

        for (_, row) in condor.do(yield_jobs(), workers, local):
            writer.writerow(row)

            out_file.flush()

if __name__ == "__main__":
    borg.script(main)

