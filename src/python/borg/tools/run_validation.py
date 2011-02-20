"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.run_validation import main

    plac.call(main)

import re
import os.path
import csv
import uuid
import itertools
import scikits.learn.linear_model
import numpy
import scipy.stats
import cargo

logger = cargo.get_logger(__name__, default_level = "INFO")

def solve_fake(solver_name, cnf_path, budget):
    """Recycle a previous solver run."""

    csv_path = cnf_path + ".rtd.csv"
    answer_map = {"": None, "True": True, "False": False}
    runs = []

    if os.path.exists(csv_path):
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)

            for (name, seed, run_budget, cost, answer) in reader:
                if name == solver_name and float(run_budget) >= budget:
                    runs.append((seed, cost, answer))

    if runs:
        (seed, cost, answer) = cargo.grab(runs)
        cost = float(cost)

        if cost > budget:
            cost = budget
            answer = None
        else:
            answer = answer_map[answer]

        return (int(seed), cost, answer)
    else:
        raise RuntimeError("no applicable runs of {0} on {1}".format(solver_name, cnf_path))

def train_random_portfolio(solvers, train_paths):
    """Build a random-action portfolio."""

    def solve_random_portfolio(cnf_path, budget):
        return cargo.grab(solvers.values())(cnf_path, budget)

    return solve_random_portfolio

def action_rates_from_runs(solver_index, budget_index, runs):
    """Build a per-action success-rate matrix from running times."""

    observed = numpy.zeros((len(solver_index), len(budget_index)))
    counts = numpy.zeros((len(solver_index), len(budget_index)), int)

    for (run_solver, _, run_budget, run_cost, run_answer) in runs:
        if run_solver in solver_index:
            solver_i = solver_index[run_solver]

            for budget in budget_index:
                budget_i = budget_index[budget]

                if run_budget >= budget:
                    counts[solver_i, budget_i] += 1

                    if run_cost <= budget and run_answer is not None:
                        observed[solver_i, budget_i] += 1.0

    return (observed, counts, observed / counts)

def get_task_run_data(task_paths):
    """Load running times associated with tasks."""

    logger.detail("loading running time data from %i files", len(task_paths))

    data = {}

    for path in task_paths:
        csv_path = "{0}.rtd.csv".format(path)
        data[path] = numpy.recfromcsv(csv_path)

    return data

def train_baseline_portfolio(solvers, train_paths):
    """Build a baseline portfolio."""

    budgets = [500]
    solver_rindex = dict(enumerate(solvers))
    solver_index = dict(map(reversed, solver_rindex.items()))
    budget_rindex = dict(enumerate(budgets))
    budget_index = dict(map(reversed, budget_rindex.items()))
    train = get_task_run_data(train_paths)
    rates = None

    for runs in train.values():
        (_, _, task_rates) = action_rates_from_runs(solver_index, budget_index, runs)

        if rates is None:
            rates = task_rates
        else:
            rates += task_rates

    rates /= len(train)
    best = solvers[solver_rindex[numpy.argmax(rates)]]

    def solve_random_portfolio(cnf_path, budget):
        return best(cnf_path, budget)

    return solve_random_portfolio

def train_classifier_portfolio(solvers, train_paths):
    """Build a classifier-based portfolio."""

    action_budget = 500
    solver_rindex = dict(enumerate(solvers))
    solver_index = dict(map(reversed, solver_rindex.items()))
    budget_rindex = dict(enumerate([action_budget]))
    budget_index = dict(map(reversed, budget_rindex.items()))

    train_xs = dict((s, []) for s in solver_index)
    train_ys = dict((s, []) for s in solver_index)

    for train_path in train_paths:
        features = numpy.recfromcsv(train_path + ".features.csv")
        (runs,) = get_task_run_data([train_path]).values()
        (_, _, rates) = action_rates_from_runs(solver_index, budget_index, runs)

        for solver in solver_index:
            successes = int(round(rates[solver_index[solver], 0] * 4))
            failures = 4 - successes

            train_xs[solver].extend([list(features.tolist())] * 4)
            train_ys[solver].extend([0] * failures + [1] * successes)

    models = {}

    for solver in solver_index:
        models[solver] = model = scikits.learn.linear_model.LogisticRegression()

        model.fit(train_xs[solver], train_ys[solver])

    def solve_classifier_portfolio(cnf_path, budget):
        features = numpy.recfromcsv(cnf_path + ".features.csv")
        scores = [(s, m.predict_proba([features.tolist()])[0, -1]) for (s, m) in models.items()]
        (name, probability) = max(scores, key = lambda (_, p): p)
        selected = solvers[name]
        (run_seed, run_cost, run_answer) = selected(cnf_path, budget - features.cost)

        return (run_seed, features.cost + run_cost, run_answer)

    return solve_classifier_portfolio

def fit_mixture_model(observed, counts):
    """Use EM to fit a discrete mixture."""

    K = 4
    components = numpy.empty((K, observed.shape[1]))
    #weights = numpy.random.random(K)
    #weights /= numpy.sum(weights)
    responsibilities = numpy.random.random((K, observed.shape[0]))
    responsibilities /= numpy.sum(responsibilities, axis = 0)

    for i in xrange(64):
        # compute new components
        for k in xrange(K):
            components[k] = numpy.sum(observed * responsibilities[k][:, None], axis = 0)
            components[k] /= numpy.sum(counts * responsibilities[k][:, None], axis = 0)

        # compute new responsibilities
        for k in xrange(K):
            mass = scipy.stats.binom.pmf(observed, counts, components[k])
            responsibilities[k] = numpy.prod(mass, axis = 1)

        responsibilities /= numpy.sum(responsibilities, axis = 0)

    weights = numpy.sum(responsibilities, axis = 1)
    weights /= numpy.sum(weights)

        #print components
        #print weights

    return (components, weights)

def train_mixture_portfolio(solvers, train_paths):
    """Build a mixture-model-based portfolio."""

    budgets = [1000]
    solver_rindex = dict(enumerate(solvers))
    solver_index = dict(map(reversed, solver_rindex.items()))
    budget_rindex = dict(enumerate(budgets))
    budget_index = dict(map(reversed, budget_rindex.items()))
    observed = numpy.empty((len(train_paths), len(solvers) * len(budgets)))
    counts = numpy.empty((len(train_paths), len(solvers) * len(budgets)))

    for (i, train_path) in enumerate(train_paths):
        (runs,) = get_task_run_data([train_path]).values()
        (runs_observed, runs_counts, _) = action_rates_from_runs(solver_index, budget_index, runs)

        observed[i] = runs_observed.flat
        counts[i] = runs_counts.flat

    (components, weights) = fit_mixture_model(observed, counts)

    def solve(cnf_path, budget):
        #print "solving", cnf_path

        actions = list(itertools.product(solvers, budgets))
        total_cost = 0.0
        history = numpy.zeros(len(actions))
        new_weights = weights

        while actions:
            # choose a solver
            probabilities = numpy.sum(components * new_weights[:, None], axis = 0)

            best_solver_i = numpy.argmax(probabilities)
            best_solver_name = solver_rindex[best_solver_i]
            best = solvers[best_solver_name]

            #print best_solver_name, probabilities, budget - total_cost

            # run it
            (_, run_cost, run_answer) = best(cnf_path, budgets[0]) # XXX
            total_cost += run_cost

            if run_answer is not None:
                return (None, total_cost, run_answer)

            history[best_solver_i] += 1.0
            new_weights = scipy.stats.binom.pmf(numpy.zeros(len(actions)), history, components)
            new_weights = numpy.prod(new_weights, axis = 1)
            new_weights *= weights
            new_weights /= numpy.sum(new_weights)

            actions = filter(lambda (_, c): c <= budget - total_cost, actions)

            # XXX obviously broken in several ways

        return (None, total_cost, None)

    return solve

core_solvers = {
    "TNM": lambda *args: solve_fake("TNM", *args),
    "march_hi": lambda *args: solve_fake("march_hi", *args),
    "cryptominisat-2.9.0": lambda *args: solve_fake("cryptominisat-2.9.0", *args),
    }
trainers = {
    #"TNM": lambda *_: lambda *args: solve_fake("TNM", *args),
    #"SATzilla2009_R": lambda *args: solve_fake("SATzilla2009_R", *args),
    "random": train_random_portfolio,
    "baseline": train_baseline_portfolio,
    "classifier": train_classifier_portfolio,
    "mixture": train_mixture_portfolio,
    }

def run_validation(name, train_paths, test_paths, budget, split):
    """Make a validation run."""

    solve = trainers[name](core_solvers, train_paths)
    successes = []

    for test_path in test_paths:
        (seed, cost, answer) = solve(test_path, budget)

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
    workers = ("submit jobs?", "option", "w", int),
    )
def main(out_path, tasks_root, workers = 0):
    """Collect validation results."""

    cargo.enable_default_logging()

    def yield_runs():
        paths = list(cargo.files_under(tasks_root, ["*.cnf"]))
        examples = int(round(min(500, len(paths)) * 0.50))

        for _ in xrange(8):
            shuffled = sorted(paths, key = lambda _ : numpy.random.rand())
            train_paths = shuffled[:examples]
            test_paths = shuffled[examples:]
            split = uuid.uuid4()

            for name in trainers:
                yield (run_validation, [name, train_paths, test_paths, 5000.0, split])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["name", "budget", "cost", "rate", "split"])

        cargo.distribute_or_labor(yield_runs(), workers, lambda _, r: writer.writerows(r))

