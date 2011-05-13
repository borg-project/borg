"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.view_write import main

    plac.call(main)

import os.path
import json
import cPickle as pickle
import numpy
import jinja2
import scikits.learn.decomposition.pca
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def sanitize(name):
    return name.replace("/", "_")

def write_category(root_path, name, category):

    # generate cluster projection
    model = category.model
    similarity_NN = numpy.dot(model._tclass_res_LN.T, model._tclass_res_LN)
    kpca = scikits.learn.decomposition.pca.KernelPCA(n_components = 2, kernel = "precomputed")
    projected_N2 = kpca.fit_transform(similarity_NN)

    # write data files
    sanitized = sanitize(name)
    data_path = os.path.join(root_path, "data", sanitized)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with open(os.path.join(data_path, "runs.json"), "w") as output_file:
        json.dump(category.table, output_file)

    with open(os.path.join(data_path, "solvers.json"), "w") as output_file:
        json.dump(category.solvers, output_file)

    with open(os.path.join(data_path, "instances.json"), "w") as output_file:
        json.dump(map(os.path.basename, category.instances), output_file)

    with open(os.path.join(data_path, "similarity.json"), "w") as output_file:
        json.dump(similarity_NN.tolist(), output_file)

    with open(os.path.join(data_path, "projection.json"), "w") as output_file:
        json.dump(projected_N2.tolist(), output_file)

    ## write symlinks
    #pseudo_path = os.path.join(root_path, "categories", sanitized)

    #if not os.path.exists(pseudo_path):
        #os.makedirs(pseudo_path)
        #os.symlink(os.path.join(root_path, "index.html"), os.path.join(pseudo_path, "index.html"))

    logger.info("wrote %s files to %s", name, data_path)

@plac.annotations(
    out_path = ("path to write visualization"),
    fit_path = ("path to visualization data"),
    )
def main(out_path, fit_path):
    """Visualize model parameters."""

    cargo.enable_default_logging()

    # load the model(s)
    logger.info("loading visualization data from %s", fit_path)

    with open(fit_path) as fit_file:
        fit = pickle.load(fit_file)

    # write data directories
    for (name, category) in fit.items():
        write_category(out_path, name, category)

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

    with open(os.path.join(out_path, "categories.json"), "w") as output_file:
        category_list = [{"name": k, "path": sanitize(k)} for k in fit.keys()]

        json.dump(category_list, output_file)

