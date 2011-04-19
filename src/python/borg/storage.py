"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import csv
import collections
import cargo

def read_train_runs(train_path):
    train_runs = collections.defaultdict(lambda: collections.defaultdict(list))
    solver_names = set()

    with cargo.openz(train_path) as train_file:
        reader = csv.reader(train_file)

        reader.next()

        for (task, solver, budget, cost, answer) in reader:
            train_runs[task][solver].append((float(budget), float(cost), bool(answer)))

            solver_names.add(solver)

    return (train_runs, solver_names)

def read_train_runs_root(train_paths):
    train_runs = collections.defaultdict(lambda: collections.defaultdict(list))
    solver_names = set()

    for train_path in train_paths:
        with cargo.openz(train_path) as train_file:
            reader = csv.reader(train_file)

            reader.next()

            for (solver, budget, cost, answer) in reader:
                train_runs[train_path][solver].append((float(budget), float(cost), bool(answer)))

                solver_names.add(solver)

    return (train_runs, solver_names)

