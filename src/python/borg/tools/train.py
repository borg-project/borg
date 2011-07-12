"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.train import main

    plac.call(main)

import cPickle as pickle
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    out_path = ("path to store solver"),
    portfolio_name = ("name of the portfolio to train"),
    solvers_path = ("path to the solvers bundle"),
    suffix = ("runs file suffix", "option"),
    tasks_roots = ("paths to training task directories"),
    )
def main(out_path, portfolio_name, solvers_path, suffix = ".runs.csv", *tasks_roots):
    """Train a solver."""

    cargo.enable_default_logging()

    # load the solvers bundle
    bundle = borg.load_solvers(solvers_path)

    # train the portfolio
    training = borg.storage.TrainingData(tasks_roots, bundle.domain, suffix = suffix)
    portfolio = borg.portfolios.named[portfolio_name](bundle, training, 50.0, 42) # XXX

    logger.info("portfolio training complete")

    # write it to disk
    with open(out_path, "w") as out_file:
        pickle.dump(portfolio, out_file, protocol = -1)

    logger.info("portfolio written to %s", out_path)

