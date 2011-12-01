"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import csv
import itertools
import numpy
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def train_model(model_name, training):
    B = 10
    T = 1

    if model_name == "mul_alpha=0.1":
        model = borg.models.MulSampler(alpha = 0.1).fit(training.solver_names, training, B, T)
    elif model_name == "mul-dir":
        model = borg.models.MulDirSampler().fit(training.solver_names, training, B, T)
    elif model_name == "mul-dirmix":
        model = borg.models.MulDirMixSampler().fit(training.solver_names, training, B, T)
    elif model_name == "mul-dirmatmix":
        model = borg.models.MulDirMatMixSampler().fit(training.solver_names, training, B, T)
    else:
        raise Exception("unrecognized model name \"{0}\"".format(model_name))

    return model

def evaluate_features(model, training, testing, feature_names):
    training = training.filter_features(feature_names)
    testing = testing.filter_features(feature_names)

    # use features
    features = []

    for instance_id in sorted(testing.feature_vectors):
        dict_ = testing.feature_vectors[instance_id]
        values = map(dict_.__getitem__, testing.common_features)

        features.append(values)

    regression = borg.regression.LinearRegression(training, model)
    weights = regression.predict(features)

    # evaluate the model
    logger.info("scoring model on %i instances", len(testing))

    score = numpy.mean(borg.models.run_data_log_probabilities(model, testing, weights))

    logger.info(
        "score given %i runs from %i instances: %f",
        training.get_run_count(),
        len(training),
        score,
        )

    return [model.name, len(feature_names), score]

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
            validation = sklearn.cross_validation.KFold(len(run_data), 5)
            (train_mask, test_mask) = iter(validation).next()
            training = run_data.masked(train_mask).collect_systematic([2])
            testing = run_data.masked(test_mask).collect_systematic([4])
            feature_counts = range(0, len(run_data.common_features) + 1)
            replications = xrange(32)
            configurations = itertools.product(feature_counts, replications)

            for model_name in experiment["model_names"]:
                model = train_model(model_name, training)
                model.name = model_name

                for (feature_count, _) in configurations:
                    feature_names = sorted(run_data.common_features, key = lambda _: numpy.random.random())
                    feature_names = feature_names[:feature_count]

                    yield (
                        evaluate_features,
                        [
                            model,
                            training,
                            testing,
                            feature_names
                            ],
                        )

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["model_name", "features", "mean_log_probability"])

        for (_, row) in condor.do(yield_jobs(), workers, local):
            writer.writerow(row)

if __name__ == "__main__":
    borg.script(main)

