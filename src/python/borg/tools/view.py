"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.view import main

    plac.call(main)

import os.path
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
    with open(solver_path) as solver_file:
        (domain, solver) = pickle.load(solver_file)

    logger.info("model loaded from %s", solver_path)

    model = solver._model
    rclass_SKB = model._rclass_SKB
    tclass_LSK = model._tclass_LSK
    tclass_weights_L = model._tclass_weights_L
    marginal_LSB = numpy.sum(tclass_LSK[..., None] * rclass_SKB[None, ...], axis = -2)

    # generate the visualization
    loader = jinja2.PackageLoader("borg.visual", "templates")
    environment = jinja2.Environment(loader = loader)

    def write_rendered(template_name, output_name, **kwargs):
        template = environment.get_template(template_name)

        with open(os.path.join(out_path, output_name), "w") as output_file:
            output_file.write(template.render(**kwargs))

    context = {
        "tclass_data_sets": marginal_LSB,
        "names": solver._solver_names,
        "budgets": solver._budgets,
        }

    write_rendered("index.html", "index.html", **context)
    write_rendered("borgview.js", "borgview.js")
    write_rendered("borgview.css", "borgview.css")

    logger.info("wrote visualization to %s", out_path)

