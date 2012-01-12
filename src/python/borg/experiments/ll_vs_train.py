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
        training = training_filtered.collect_independent(mixture)
    else:
        training = training_filtered.collect_systematic(mixture)

    # build the model
    bins = 30
    samplers = {
        "mul_alpha=0.1": borg.models.MulSampler(alpha = 0.1),
        "mul-dir": borg.models.MulDirSampler(),
        "mul-dirmix": borg.models.MulDirMixSampler(),
        "mul-dirmatmix": borg.models.MulDirMatMixSampler(),
        }

    if model_name in samplers:
        model = \
            borg.models.mean_posterior(
                samplers[model_name],
                training.solver_names,
                training,
                bins = bins,
                chains = 1,
                samples_per_chain = 1,
                )
    elif model_name == "log-normal":
        estimator = borg.models.LogNormalMixEstimator()
        model = estimator(training, bins)
    else:
        raise Exception("unrecognized model name \"{0}\"".format(model_name))

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

    mixture_description = \
        "{0} ({1})".format(
            "/".join(map(str, mixture)),
            "Ind." if independent else "Sys.",
            )

    return [model_name, mixture_description, instance_count, score]

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
            validation = sklearn.cross_validation.KFold(len(run_data), 10)
            max_instance_count = numpy.floor(0.9 * len(run_data)) - 10
            #instance_counts = map(int, map(round, numpy.r_[10:max_instance_count:16j]))
            instance_counts = [int(max_instance_count / 2)]

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

