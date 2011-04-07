"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.view import main

    plac.call(main)

import jinja2
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    out_path = ("path to write visualization"),
    domain_name = ("name of the problem domain"),
    tasks_roots = ("paths to training task directories"),
    )
def main(out_path, domain_name, *tasks_roots):
    """Visualize model parameters."""

    cargo.enable_default_logging()

    # find the training tasks
    domain = borg.get_domain(domain_name)
    train_paths = []

    for tasks_root in tasks_roots:
        train_paths.extend(cargo.files_under(tasks_root, domain.extensions))

    logger.info("using %i tasks for training", len(train_paths))

    # fit the model
    solver = BilevelPortfolio(domain, train_paths, 100.0, 50)

    logger.info("portfolio training complete")

    model = solver._model
    rclass_SKB = model._rclass_SKB
    tclass_LSK = model._tclass_LSK

    # generate the visualization
    loader = jinja2.PackageLoader("borg.tools.visual", "templates")
    environment = jinja2.Environment(loader = loader)
    template = environment.get_template("index.html")

    print template.render()

