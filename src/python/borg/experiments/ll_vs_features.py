"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import csv
import itertools
import numpy
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def evaluate_features(model, testing, feature_names):
    # use features
    if len(feature_names) > 0:
        # train the prediction method
        feature_mask = numpy.array([f in feature_names for f in testing.common_features])
        masked_model = model.with_new(features = model.features[:, feature_mask])
        regression = borg.regression.NearestRTDRegression(masked_model)

        # then test it
        test_features = testing.filter_features(feature_names).to_features_array()
        weights = regression.predict(None, test_features)
    else:
        weights = numpy.tile(numpy.exp(model.log_weights), (len(testing), 1))

    # evaluate the model
    logger.info("scoring %s on %i instances", model.name, len(testing))

    log_probabilities = borg.models.run_data_log_probabilities(model, testing, weights)

    return [
        [model.name, len(feature_names), "mean_log_probability", numpy.mean(log_probabilities)],
        [model.name, len(feature_names), "median_log_probability", numpy.median(log_probabilities)],
        ]

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
            validation = sklearn.cross_validation.KFold(len(run_data), 5, indices = False)
            (train_mask, test_mask) = iter(validation).next()
            training = run_data.masked(train_mask).collect_systematic([2])
            testing = run_data.masked(test_mask).collect_systematic([4])
            feature_counts = range(0, len(run_data.common_features) + 1, 2)
            replications = xrange(32)
            parameters = list(itertools.product(feature_counts, replications))

            for model_name in experiment["model_names"]:
                model = borg.experiments.common.train_model(model_name, training)
                model.name = model_name

                for (feature_count, _) in parameters:
                    shuffled_names = sorted(run_data.common_features, key = lambda _: numpy.random.random())
                    selected_names = sorted(shuffled_names[:feature_count])

                    yield (
                        evaluate_features,
                        [
                            model,
                            testing,
                            selected_names,
                            ],
                        )

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["model_name", "features", "score_name", "score"])

        for (_, rows) in condor.do(yield_jobs(), workers, local):
            writer.writerows(rows)

            out_file.flush()

if __name__ == "__main__":
    borg.script(main)

