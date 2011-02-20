"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.train import main

    plac.call(main)

import os.path
import cPickle as pickle
import borg

@plac.annotations(
    out_path = ("path to store solver", "positional", None, os.path.abspath),
    tasks_root = ("path to training tasks", "positional", None, os.path.abspath),
    )
def main(out_path, name, tasks_root):
    """Train a solver."""

    cargo.enable_default_logging()

    train_paths = list(cargo.files_under(tasks_root, ["*.cnf"]))

    logger.info("using %i tasks for training", len(train_paths))

    solvers = borg.solvers.named
    trainers = borg.portfolios.named
    solver = trainers[name](solvers, train_paths)

    with open(out_path, "w") as out_file:
        pickle.dump(solver, out_file, protocol = -1)

