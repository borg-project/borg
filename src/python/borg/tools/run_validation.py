"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.run_validation import main

    plac.call(main)

import os.path
import csv
import uuid
import itertools
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def solve_fake(solver_name, cnf_path, budget):
    """Recycle a previous solver run."""

    csv_path = cnf_path + ".rtd.csv"
    answer_map = {"": None, "True": True, "False": False}
    runs = []

    if os.path.exists(csv_path):
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)

            for (name, _, run_budget, cost, answer) in reader:
                if name == solver_name and float(run_budget) >= budget:
                    runs.append((cost, answer))

    if runs:
        (cost, answer) = cargo.grab(runs)
        cost = float(cost)

        if cost > budget:
            cost = budget
            answer = None
        else:
            answer = answer_map[answer]

        return (cost, answer)
    else:
        raise RuntimeError("no applicable runs of {0} on {1}".format(solver_name, cnf_path))

def fake_solver(solver_name):
    return lambda *args: solve_fake(solver_name, *args)

subsolvers = ["TNM", "march_hi", "gnovelty+2", "hybridGM3", "adaptg2wsat++"]

core_solvers = dict(zip(subsolvers, map(fake_solver, subsolvers)))

portfolios = borg.portfolios.named.copy()

#portfolios["SATzilla2009_R"] = lambda *_: lambda *args: solve_fake("SATzilla2009_R", *args)

def run_validation(name, train_paths, test_paths, budget, split):
    """Make a validation run."""

    solver = portfolios[name](core_solvers, train_paths)
    successes = []

    for test_path in test_paths:
        (cost, answer) = solver(test_path, budget)

        if answer is not None:
            successes.append(cost)

    rate = float(len(successes)) / len(test_paths)

    logger.info("method %s had final success rate %.2f", name, rate)

    return \
        zip(
            itertools.repeat(name),
            itertools.repeat(budget),
            sorted(successes),
            numpy.arange(len(successes) + 1.0) / len(test_paths),
            itertools.repeat(split),
            )

@plac.annotations(
    out_path = ("path to results file", "positional", None, os.path.abspath),
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    runs = ("number of runs", "option", "r", int),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(out_path, tasks_root, runs = 16, workers = 0):
    """Collect validation results."""

    cargo.enable_default_logging()

    cargo.get_logger("borg.portfolios", level = "DETAIL")

    def yield_runs():
        paths = list(cargo.files_under(tasks_root, ["*.cnf"]))
        examples = int(round(len(paths) * 0.50))

        for _ in xrange(runs):
            shuffled = sorted(paths, key = lambda _ : numpy.random.rand())
            train_paths = shuffled[:examples]
            test_paths = shuffled[examples:]
            split = uuid.uuid4()

            for name in portfolios:
                yield (run_validation, [name, train_paths, test_paths, 5000.0, split])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["name", "budget", "cost", "rate", "split"])

        cargo.distribute_or_labor(yield_runs(), workers, lambda _, r: writer.writerows(r))

