"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import copy
import numpy
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def infer_distributions(run_data, model_name, instance, exclude):
    """Compute model predictions on every instance."""

    # customize the run data
    filtered_data = copy.deepcopy(run_data)

    filtered_runs = filter(lambda r: r.solver != exclude, filtered_data.run_lists[instance])

    filtered_data.run_lists[instance] = filtered_runs

    # sample from the model posterior
    model = borg.experiments.common.train_model(model_name, filtered_data, bins = 8)

    # summarize the samples
    (M, S, B) = model.log_masses.shape

    n = sorted(filtered_data.run_lists).index(instance)

    true = run_data.to_bins_array(run_data.solver_names, B = 8)[n].astype(float)
    true /= numpy.sum(true, axis = -1)[:, None] + 1e-16

    observed = filtered_data.to_bins_array(run_data.solver_names, B = 8)[n].astype(float)
    observed /= numpy.sum(observed, axis = -1)[:, None] + 1e-16

    mask = model.names == instance
    log_predicted_all = model.log_weights[mask][:, None, None] + model.log_masses[mask, :, :]
    predicted = numpy.sum(numpy.exp(log_predicted_all), axis = 0)
    predicted /= numpy.sum(predicted, axis = -1)[..., None]

    def yield_rows():
        for s in xrange(S):
            solver_name = run_data.solver_names[s]

            for b in xrange(B):
                yield (model_name, instance, solver_name, b, predicted[s, b])
                yield ("observed", instance, solver_name, b, observed[s, b])
                yield ("true", instance, solver_name, b, true[s, b])

    return list(yield_rows())

@borg.annotations(
    out_path = ("results output path"),
    bundle = ("path to pre-recorded runs", "positional", None, os.path.abspath),
    experiments = ("path to experiments JSON", "positional", None, borg.util.load_json),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, bundle, experiments, workers = 0, local = False):
    """Write the actual output of multiple models."""

    def yield_jobs():
        run_data = borg.storage.RunData.from_bundle(bundle)

        for experiment in experiments:
            yield (
                infer_distributions,
                [
                    run_data,
                    experiment["model_name"],
                    experiment["instance"],
                    experiment["exclude"],
                    ],
                )

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["model_name", "instance", "solver", "bin", "probability"])

        for (_, row) in condor.do(yield_jobs(), workers, local):
            writer.writerows(row)

if __name__ == "__main__":
    borg.script(main)

