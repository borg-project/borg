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

def fake_resume_solver(run_cost, run_answer, position):
    """Return a resume call for a fake run."""

    def resume(_, budget):
        new_position = budget + position

        if new_position >= run_cost:
            return (run_cost - position, run_answer, None)
        else:
            return (budget, None, fake_resume_solver(run_cost, run_answer, new_position))

    return resume

def solve_fake(solver_name, cnf_path, budget):
    """Recycle a previous solver run."""

    # gather candidate runs
    runs = []
    answer_map = {"": None, "True": True, "False": False}

    with open(cnf_path + ".rtd.csv") as csv_file:
        reader = csv.reader(csv_file)

        for (name, _, run_budget, run_cost, run_answer) in reader:
            if name == solver_name:
                run_cost = float(run_cost)
                run_budget = float(run_budget)

                if run_budget >= 6000.0:
                    runs.append((run_cost, answer_map[run_answer]))

    # return one of them at random
    (run_cost, run_answer) = cargo.grab(runs)

    if run_cost > budget:
        return (budget, None, fake_resume_solver(run_cost, run_answer, budget))
    else:
        return (run_cost, run_answer, None)

def fake_solver(solver_name):
    return lambda *args: solve_fake(solver_name, *args)

subsolvers = [
    "kcnfs-2006",
    "hybridGM3",
    "NCVWr",
    "gnovelty+2",
    "iPAWS",
    "adaptg2wsat2009++",
    "TNM",
    "march_hi",
    "FH",
    ]

core_solvers = dict(zip(subsolvers, map(fake_solver, subsolvers)))

portfolios = borg.portfolios.named.copy()

#portfolios["SATzilla2009_R"] = lambda *_: lambda *args: solve_fake("SATzilla2009_R", *args)

def run_validation(name, train_paths, test_paths, budget, split):
    """Make a validation run."""

    solver = portfolios[name](core_solvers, train_paths)
    successes = []

    for test_path in test_paths:
        (cost, answer, _) = solver(test_path, budget)

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
    tests_root = ("optional separate test set", "positional", None, os.path.abspath),
    runs = ("number of runs", "option", "r", int),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(out_path, tasks_root, tests_root = None, runs = 16, workers = 0):
    """Collect validation results."""

    cargo.enable_default_logging()

    cargo.get_logger("borg.portfolios", level = "DETAIL")

    def yield_runs():
        paths = list(cargo.files_under(tasks_root, ["*.cnf"]))
        examples = int(round(len(paths) * 0.50))

        if tests_root is not None:
            tests_root_paths = list(cargo.files_under(tests_root, ["*.cnf"]))

        for _ in xrange(runs):
            shuffled = sorted(paths, key = lambda _ : numpy.random.rand())
            train_paths = shuffled[:examples]

            if tests_root is None:
                test_paths = shuffled[examples:]
            else:
                test_paths = tests_root_paths

            split = uuid.uuid4()

            for name in portfolios:
                yield (run_validation, [name, train_paths, test_paths, 5000.0, split])

    with open(out_path, "w") as out_file:
    #with open(out_path, "a") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["name", ">budget", "cost", "rate", "split"])

        cargo.distribute_or_labor(yield_runs(), workers, lambda _, r: writer.writerows(r))

