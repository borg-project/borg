"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import uuid
import numpy
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def evaluate_split(run_data, alpha, split, train_mask, test_mask):
    """Evaluate a model on a train/test split."""

    training = run_data.masked(train_mask).collect_systematic([4])
    testing = run_data.masked(test_mask).collect_systematic([4])
    model = borg.models.MulEstimator(alpha = alpha)(training, 10, training)
    score = numpy.mean(borg.models.run_data_log_probabilities(model, testing))

    logger.info(
        "score at alpha = %.2f given %i runs from %i instances: %f",
        alpha,
        training.get_run_count(),
        len(training),
        score,
        )

    return [alpha, len(training), split, score]

@borg.annotations(
    out_path = ("results output path"),
    bundle = ("path to pre-recorded runs", "positional", None, os.path.abspath),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, bundle, workers = 0, local = False):
    """Evaluate the pure multinomial model over a range of smoothing values."""

    def yield_jobs():
        run_data = borg.storage.RunData.from_bundle(bundle)
        validation = sklearn.cross_validation.KFold(len(run_data), 10, indices = False)

        for (train_mask, test_mask) in validation:
            split = uuid.uuid4()
            alphas = numpy.r_[1e-8:1e-1:64j]

            for alpha in alphas:
                yield (evaluate_split, [run_data, alpha, split, train_mask, test_mask])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["alpha", "instances", "split", "mean_log_probability"])

        for (_, row) in condor.do(yield_jobs(), workers, local):
            writer.writerow(row)

            out_file.flush()

if __name__ == "__main__":
    borg.script(main)

