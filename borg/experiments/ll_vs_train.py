"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import csv
import numpy
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def evaluate_split(run_data, model_name, mixture, independent, instance_count, train_mask, test_mask):
    """Evaluate a model on a train/test fold."""

    testing = run_data.masked(test_mask).collect_systematic([4])
    training_all = run_data.masked(train_mask)
    training_ids = sorted(training_all.ids, key = lambda _: numpy.random.rand())
    training_filtered = training_all.filter(*training_ids[:instance_count])

    if independent:
        training = training_filtered.collect_independent(mixture).only_nonempty()
    else:
        training = training_filtered.collect_systematic(mixture).only_nonempty()

    # build the model
    bins = 10
    estimators = {
        "mul_alpha=0.1": borg.models.MulEstimator(alpha = 2e-2),
        "mul-dir": borg.models.MulDirEstimator(),
        "mul-dirmix": borg.models.MulDirMixEstimator(K = 4),
        "mul-dirmatmix": borg.models.MulDirMatMixEstimator(K = 32),
        }
    model = estimators[model_name](training, bins, training_all)

    # evaluate the model
    logger.info("scoring model on %i instances", len(testing))

    score = numpy.mean(borg.models.run_data_log_probabilities(model, testing))

    logger.info(
        "score of model %s given %i runs from %i instances: %f",
        model_name,
        training.get_run_count(),
        len(training),
        score,
        )

    description = "{0} ({1})".format(mixture, "Sep." if independent else "Sys.")

    return [model_name, description, instance_count, score]

@borg.annotations(
    out_path = ("results output path"),
    experiments = ("path to experiments JSON", "positional", None, borg.util.load_json),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, experiments, workers = 0, local = False):
    """Run the specified model evaluations."""

    logger.info("running %i experiments", len(experiments))

    get_run_data = borg.util.memoize(borg.storage.RunData.from_bundle)

    def yield_jobs():
        for experiment in experiments:
            logger.info("preparing experiment: %s", experiment)

            run_data = get_run_data(experiment["run_data"])
            validation = sklearn.cross_validation.ShuffleSplit(len(run_data), 32, test_fraction = 0.1, indices = False)
            max_instance_count = numpy.floor(0.9 * len(run_data)) - 10
            instance_counts = map(int, map(round, numpy.r_[10:max_instance_count:24j]))

            for (train_mask, test_mask) in validation:
                for instance_count in instance_counts:
                    yield (
                        evaluate_split,
                        [
                            run_data,
                            experiment["model_name"],
                            experiment["mixture"],
                            experiment["independent"],
                            instance_count,
                            train_mask,
                            test_mask,
                            ],
                        )

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["model_name", "sampling", "instances", "mean_log_probability"])

        for (_, row) in condor.do(yield_jobs(), workers, local):
            writer.writerow(row)

            out_file.flush()

if __name__ == "__main__":
    borg.script(main)

