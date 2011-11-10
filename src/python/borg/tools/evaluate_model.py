"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.evaluate_model import main

    plac.call(main)

import os.path
import csv
import uuid
import numpy
import sklearn
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def score_model_log_probability(model, B, testing):
    logger.info("scoring model on %i instances", len(testing))

    counts = testing.to_bins_array(testing.solver_names, B)
    log_probabilities = borg.models.sampled_pmfs_log_pmf(model.log_masses, counts)
    lps_per_instance = numpy.logaddexp.reduce(log_probabilities, axis = 0)

    #return lps_per_instance
    return numpy.mean(lps_per_instance)

def evaluate_split(model_name, split, training, testing):
    """Evaluate a model on a train/test split."""

    #if cargo.get_task() is not None:
        #borg.statistics.set_prng_seeds(hash(cargo.get_task().key))
    borg.statistics.set_prng_seeds()

    # build the model
    B = 10
    T = 1

    if model_name == "mul_alpha=0.1":
        model = borg.models.Mul_ModelFactory(alpha = 0.1).fit(training.solver_names, training, B, T)
    elif model_name == "mul_alpha=1.0":
        model = borg.models.Mul_ModelFactory(alpha = 1.0).fit(training.solver_names, training, B, T)
    elif model_name == "mul_alpha=100.0":
        model = borg.models.Mul_ModelFactory(alpha = 100.0).fit(training.solver_names, training, B, T)
    elif model_name == "mul-dir":
        model = borg.models.Mul_Dir_ModelFactory().fit(training.solver_names, training, B, T)
    elif model_name == "mul-dirmix":
        model = borg.models.Mul_DirMix_ModelFactory().fit(training.solver_names, training, B, T)
    else:
        raise Exception("unrecognized model name \"{0}\"".format(model_name))

    # evaluate the model
    #scores = score_model_log_probability(model, B, testing)

    #return [[model_name, len(training), split, score] for score in scores]

    score = score_model_log_probability(model, B, testing)

    logger.info(
        "score of model %s given %i runs from %i instances: %f",
        model_name,
        training.get_run_count(),
        len(training),
        score,
        )

    return [model_name, len(training), split, score]

@plac.annotations(
    out_path = ("results output path"),
    model_name = ("name of the model to train"),
    bundle = ("path to pre-recorded runs", "positional", None, os.path.abspath),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, model_name, bundle, workers = 0, local = False):
    """Evaluate the specified model."""

    cargo.enable_default_logging()

    def yield_jobs():
        run_data = borg.storage.RunData.from_bundle(bundle)
        validation = sklearn.cross_validation.KFold(len(run_data), 10)
        restarts_mixture = [4, 1]

        logger.info("yielding jobs for restarts mixture %s", restarts_mixture)

        for (train_mask, test_mask) in validation:
            split = uuid.uuid4()
            training = run_data.masked(train_mask)
            testing = run_data.masked(test_mask)
            instance_counts = map(int, map(round, numpy.r_[2:len(training):16j]))
            training_ids = sorted(training.ids, key = lambda _: numpy.random.rand())

            for instance_count in instance_counts:
                subset = training.filter(*training_ids[:instance_count]).collect(restarts_mixture)

                yield (evaluate_split, [model_name, split, subset, testing])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["model_name", "instances", "split", "mean_log_probability"])

        cargo.do_or_distribute(yield_jobs(), workers, lambda _, r: writer.writerow(r), local)

