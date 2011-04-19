"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.train import main

    plac.call(main)

import csv
import cPickle as pickle
import collections
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    out_path = ("path to store solver"),
    portfolio_name = ("name of the portfolio to train"),
    domain_name = ("name of the problem domain"),
    train_path = ("path to training data"),
    )
def main(out_path, portfolio_name, domain_name, train_path):
    """Train a solver."""

    cargo.enable_default_logging()

    # find the training tasks
    (train_runs, solver_names) = borg.read_train_runs(train_path)

    logger.info("loaded %i runs for training", len(train_runs))

    # train the portfolio
    #domain = borg.get_domain(domain_name)
    import borg.tools.run_validation
    domain = borg.tools.run_validation.FakeDomain(borg.get_domain(domain_name), solver_names)
    solver = borg.portfolios.named[portfolio_name](domain, train_runs, 100.0, 50)

    logger.info("portfolio training complete")

    # write it to disk
    with open(out_path, "w") as out_file:
        pickle.dump((domain, solver), out_file, protocol = -1)

    logger.info("portfolio written to %s", out_path)

