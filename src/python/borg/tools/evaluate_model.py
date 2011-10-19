"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.evaluate_model import main

    plac.call(main)

import os.path
import csv
import uuid
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def score_model_log_probability(model, B, testing):
    logger.info("scoring model on %i instances", len(testing))

    counts = testing.to_bins_array(testing.solver_names, B)
    log_probabilities = borg.models.sampled_pmfs_log_pmf(model.log_masses, counts)

    print model.log_masses[0]

    return numpy.mean(log_probabilities)

def evaluate_split(model_name, split, training, testing):
    # build the model
    B = 10
    T = 1

    if model_name == "mul":
        model = borg.models.Mul_ModelFactory().fit(training.solver_names, training, B, T)
    elif model_name == "mul-dir":
        model = borg.models.Mul_Dir_ModelFactory().fit(training.solver_names, training, B, T)
    elif model_name == "mul-dirmix":
        model = borg.models.Mul_DirMix_ModelFactory().fit(training.solver_names, training, B, T)
    else:
        raise Exception("unrecognized model name")

    # evaluate the model
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

        for _ in xrange(6):
            split = uuid.uuid4()
            (training, testing) = run_data.split(0.5)
            instance_counts = numpy.r_[1:len(training):16]
            #instance_counts = [1, 16]
            training_ids = sorted(training.ids, key = lambda _: numpy.random.rand())

            for instance_count in instance_counts:
                subset = training.filter(*training_ids[:instance_count]).collect([4])

                yield (evaluate_split, [model_name, split, subset, testing])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["model_name", "instances", "split", "mean_log_probability"])

        cargo.do_or_distribute(yield_jobs(), workers, lambda _, r: writer.writerow(r), local)

