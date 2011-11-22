"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import uuid
import numpy
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def evaluate_split(alpha, split, training, testing):
    """Evaluate a model on a train/test split."""

    # build the model
    B = 10
    T = 1

    model = borg.models.MulSampler(alpha = alpha).fit(training.solver_names, training, B, T)

    # evaluate the model
    log_probabilities = borg.models.run_data_log_probabilities(model, B, testing)
    score = numpy.mean(log_probabilities)

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
        validation = sklearn.cross_validation.KFold(len(run_data), 10)

        for (train_mask, test_mask) in validation:
            split = uuid.uuid4()
            training = run_data.masked(train_mask)
            testing = run_data.masked(test_mask)
            alphas = numpy.r_[1e-4:1:128j]

            for alpha in alphas:
                yield (evaluate_split, [alpha, split, training, testing])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["alpha", "instances", "split", "mean_log_probability"])

        condor.do(yield_jobs(), workers, lambda _, r: writer.writerow(r), local)

if __name__ == "__main__":
    borg.script(main)

