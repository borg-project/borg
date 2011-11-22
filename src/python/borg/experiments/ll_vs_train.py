"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import uuid
import json
import numpy
import sklearn
import borg

logger = borg.get_logger(__name__)

def evaluate_split(model_name, mixture, split, training, testing):
    """Evaluate a model on a train/test split."""

    # build the model
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

    # evaluate the model
    logger.info("scoring model on %i instances", len(testing))

    counts = testing.to_bins_array(testing.solver_names, B)
    log_probabilities = borg.models.sampled_pmfs_log_pmf(model.log_masses, counts)
    lps_per_instance = numpy.logaddexp.reduce(log_probabilities, axis = 0)
    score = numpy.mean(lps_per_instance)

    logger.info(
        "score of model %s given %i runs from %i instances: %f",
        model_name,
        training.get_run_count(),
        len(training),
        score,
        )

    return [model_name, len(training), split, score]
    #return [model_name, ",".join(map(str, mixture)), len(training), split, score]

@borg.annotations(
    out_path = ("results output path"),
    model_name = ("name of the model to train"),
    mixture = ("restarts mixture", "positional", None, json.loads),
    bundle = ("path to pre-recorded runs", "positional", None, os.path.abspath),
    independent = ("restarts are independent per-solver?", "flag"),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, model_name, bundle, mixture, independent = False, workers = 0, local = False):
    """Evaluate the specified model."""

    def yield_jobs():
        run_data = borg.storage.RunData.from_bundle(bundle)
        validation = sklearn.cross_validation.KFold(len(run_data), 10)

        logger.info("yielding jobs for restarts mixture %s", mixture)

        for (train_mask, test_mask) in validation:
            split = uuid.uuid4()
            testing = run_data.masked(test_mask)
            training_all = run_data.masked(train_mask)
            instance_counts = map(int, map(round, numpy.r_[10:len(training_all):16j]))
            training_ids = sorted(training_all.ids, key = lambda _: numpy.random.rand())

            for instance_count in instance_counts:
                training_filtered = training_all.filter(*training_ids[:instance_count])
                
                if independent:
                    training = training_filtered.collect_independent(mixture)
                else:
                    training = training_filtered.collect_systematic(mixture)

                yield (evaluate_split, [model_name, mixture, split, training, testing])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["model_name", "instances", "split", "mean_log_probability"])

        borg.do(yield_jobs(), workers, lambda _, r: writer.writerow(r), local)

if __name__ == "__main__":
    borg.script(main)

