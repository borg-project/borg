"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import uuid
import numpy
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def evaluate_split(model_name, K, split, training, testing):
    """Evaluate a model on a train/test split."""

    # build the model
    if model_name == "mul-dirmix":
        sampler = borg.models.MulDirMixSampler(K = K)
    elif model_name == "mul-dirmatmix":
        sampler = borg.models.MulDirMatMixSampler(K = K)
    else:
        raise ValueError("unrecognized model name {0}".format(model_name))

    B = 10
    model = sampler.fit(training.solver_names, training, B = B, T = 1)

    # evaluate the model
    log_probabilities = borg.models.run_data_log_probabilities(model, B, testing)
    score = numpy.mean(log_probabilities)

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

    borg.get_logger("condor.labor", level = "NOTSET")

    def yield_jobs():
        run_data = borg.storage.RunData.from_bundle(bundle)
        validation = sklearn.cross_validation.KFold(len(run_data), 10)

        for (train_mask, test_mask) in validation:
            split = uuid.uuid4()
            training = run_data.masked(train_mask)
            testing = run_data.masked(test_mask)
            Ks = range(1, 64, 2)

            for K in Ks:
                for model_name in ["mul-dirmix", "mul-dirmatmix"]:
                    yield (evaluate_split, [model_name, K, split, training, testing])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["model_name", "components", "instances", "split", "mean_log_probability"])

        condor.do(yield_jobs(), workers, lambda _, r: writer.writerow(r), local)

if __name__ == "__main__":
    borg.script(main)

