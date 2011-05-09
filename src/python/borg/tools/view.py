"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.view import main

    plac.call(main)

import os.path
import json
import cPickle as pickle
import numpy
import jinja2
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    out_path = ("path to write visualization"),
    solver_path = ("path to the trained solver"),
    )
def main(out_path, solver_path):
    """Visualize model parameters."""

    cargo.enable_default_logging()

    # load the model
    logger.info("loading model from %s", solver_path)

    with open(solver_path) as solver_file:
        (domain, solver) = pickle.load(solver_file)

    # load the training data
    runs = []
    solvers = set()

    logger.info("loading data from %i paths", len(solver._train_paths))

    for task_path in solver._train_paths:
        (task_runs,) = borg.portfolios.get_task_run_data([task_path]).values()
        task_runs_list = []
        task_solvers = set() # XXX support multiple runs

        for (run_solver, _, run_budget, run_cost, run_answer) in task_runs.tolist():
            # XXX support multiple runs
            if run_solver not in task_solvers:
                task_runs_list.append({
                    "solver": run_solver,
                    "budget": run_budget,
                    "cost": run_cost,
                    "answer": run_answer,
                    })

                solvers.add(run_solver)
                task_solvers.add(run_solver)

        runs.append({
            "instance": os.path.basename(task_path),
            "runs": task_runs_list,
            })

    solvers = list(solvers)

    # extract the model parameters
    model = solver._model
    rclass_SKB = model._rclass_SKB
    tclass_LSK = model._tclass_LSK
    tclass_res_LN = model._tclass_res_LN
    tclass_weights_L = model._tclass_weights_L
    marginal_LSB = numpy.sum(tclass_LSK[..., None] * rclass_SKB[None, ...], axis = -2)
    similarity_NN = numpy.dot(tclass_res_LN.T, tclass_res_LN)

    # generate the visualization
    loader = jinja2.PackageLoader("borg.visual", "templates")
    environment = jinja2.Environment(loader = loader)

    def write_rendered(template_name, output_name, **kwargs):
        template = environment.get_template(template_name)

        with open(os.path.join(out_path, output_name), "w") as output_file:
            output_file.write(template.render(**kwargs))

    write_rendered("index.html", "index.html")
    write_rendered("borgview.css", "borgview.css")
    write_rendered("borgview.js", "borgview.js")

    with open(os.path.join(out_path, "data/runs.json"), "w") as output_file:
        json.dump(runs, output_file)

    with open(os.path.join(out_path, "data/solvers.json"), "w") as output_file:
        json.dump(solvers, output_file)

    with open(os.path.join(out_path, "data/instances.json"), "w") as output_file:
        json.dump(map(os.path.basename, solver._train_paths), output_file)

    with open(os.path.join(out_path, "data/similarity.json"), "w") as output_file:
        json.dump(similarity_NN.tolist(), output_file)

    logger.info("wrote visualization to %s", out_path)

