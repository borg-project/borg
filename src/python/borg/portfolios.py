"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import uuid
import resource
import itertools
import contextlib
import multiprocessing
import numpy
import scipy.stats
import scikits.learn.linear_model
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@contextlib.contextmanager
def fpe_handling(**kwargs):
    previous = numpy.seterr(**kwargs)

    yield

    numpy.seterr(**previous)

def get_task_run_data(task_paths):
    """Load running times associated with tasks."""

    data = {}

    for path in task_paths:
        csv_path = "{0}.rtd.csv".format(path)
        data[path] = numpy.recfromcsv(csv_path, usemask = True)

    return data

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

class RandomPortfolio(object):
    """Random-action portfolio."""

    def __init__(self, solvers, train_paths):
        self._solvers = solvers
    
    def __call__(self, cnf_path, budget, cores):
        queue = multiprocessing.Queue()
        solvers = []

        for _ in xrange(cores):
            solver_id = uuid.uuid4()
            solver = cargo.grab(self._solvers.values())(cnf_path, queue, solver_id)

            solvers.append(solver)

        try:
            for solver in solvers:
                solver.go(budget)

            while True:
                (solver_id, run_cost, answer) = queue.get()

                if answer is not None:
                    return (run_cost, answer)
        finally:
            for solver in solvers:
                solver.die()

class BaselinePortfolio(object):
    """Baseline portfolio."""

    def __init__(self, solvers, train_paths):
        budgets = [500]
        solver_rindex = dict(enumerate(solvers))
        solver_index = dict(map(reversed, solver_rindex.items()))
        budget_rindex = dict(enumerate(budgets))
        budget_index = dict(map(reversed, budget_rindex.items()))
        train = get_task_run_data(train_paths)
        rates = None

        for runs in train.values():
            (_, _, task_rates) = action_rates_from_runs(solver_index, budget_index, runs.tolist())

            if rates is None:
                rates = task_rates
            else:
                rates += task_rates

        rates /= len(train)

        self._best = solvers[solver_rindex[numpy.argmax(rates)]]

    def __call__(self, cnf_path, budget):
        return self._best(cnf_path, budget)

class OraclePortfolio(object):
    """Oracle-approximate portfolio."""

    def __init__(self, solvers, train_paths):
        self._solvers = solvers

    def __call__(self, cnf_path, budget):
        solver_rindex = dict(enumerate(self._solvers))
        solver_index = dict(map(reversed, solver_rindex.items()))
        (runs,) = get_task_run_data([cnf_path]).values()
        (_, _, rates) = action_rates_from_runs(solver_index, {budget: 0}, runs.tolist())
        best = self._solvers[solver_rindex[numpy.argmax(rates)]]

        return best(cnf_path, budget)

def knapsack_plan(rates, solver_rindex, budgets, remaining):
    """
    Generate an optimal plan for a static probability table.

    Budgets *must* be in ascending order and *must* be spaced by a fixed
    interval from zero.
    """

    # find the index of our starting budget
    start = None

    for (b, budget) in enumerate(budgets):
        if budget > remaining:
            break
        else:
            start = b

    if start is None:
        return []

    # generate the values table and associated policy
    with fpe_handling(divide = "ignore"):
        log_rates = numpy.log(1.0 - rates)

    (nsolvers, nbudgets) = log_rates.shape
    values = numpy.zeros(nbudgets + 1)
    policy = {}

    for b in xrange(1, nbudgets + 1):
        region = log_rates[:, :b] + values[None, b - 1::-1]
        i = numpy.argmin(region)
        (s, c) = numpy.unravel_index(i, region.shape)

        values[b] = region[s, c]
        policy[b] = (s, c)

    # build a plan from the policy
    plan = []
    b = start + 1

    while b > 0:
        (_, c) = action = policy[b]
        b -= c + 1

        plan.append(action)

    # heuristically reorder the plan
    reordered = sorted(plan, key = lambda (s, c): -rates[s, c] / budgets[c])

    # translate and return it
    return [(solver_rindex[s], budgets[c]) for (s, c) in reordered]

class UberOraclePortfolio(object):
    """Oracle-approximate planning portfolio."""

    def __init__(self, solvers, train_paths):
        self._solvers = solvers
        self._budgets = [(b + 1) * 100.0 for b in xrange(50)]
        self._solver_rindex = dict(enumerate(self._solvers))
        self._solver_index = dict(map(reversed, self._solver_rindex.items()))
        self._budget_rindex = dict(enumerate(self._budgets))
        self._budget_index = dict(map(reversed, self._budget_rindex.items()))

    def __call__(self, cnf_path, budget):
        logger.debug("solving %s", cnf_path)

        # grab known run data
        (runs,) = get_task_run_data([cnf_path]).values()
        (_, _, rates) = action_rates_from_runs(self._solver_index, self._budget_index, runs.tolist())

        # make a plan
        plan = knapsack_plan(rates, self._solver_rindex, self._budgets, budget)

        # follow through
        total_cost = 0.0
        answer = None

        for (name, cost) in plan:
            solver = self._solvers[name]
            (run_cost, answer, _) = solver(cnf_path, cost)

            b = self._budget_index[cost]
            logger.debug(
                "ran %s@%i with %is; yielded %s (p = %f)",
                name,
                cost,
                budget - total_cost,
                answer,
                rates[self._solver_index[name], b],
                )

            total_cost += run_cost

            if answer is not None:
                break

        return (total_cost, answer, None)

class ClassifierPortfolio(object):
    """Classifier-based portfolio."""

    def __init__(self, solvers, train_paths):
        self._solvers = solvers

        action_budget = 5000
        solver_rindex = dict(enumerate(solvers))
        solver_index = dict(map(reversed, solver_rindex.items()))
        budget_rindex = dict(enumerate([action_budget]))
        budget_index = dict(map(reversed, budget_rindex.items()))

        # prepare training data
        train_xs = dict((s, []) for s in solver_index)
        train_ys = dict((s, []) for s in solver_index)

        for train_path in train_paths:
            features = numpy.recfromcsv(train_path + ".features.csv")
            (runs,) = get_task_run_data([train_path]).values()
            (_, _, rates) = action_rates_from_runs(solver_index, budget_index, runs.tolist())

            for solver in solver_index:
                successes = int(round(rates[solver_index[solver], 0] * 4))
                failures = 4 - successes

                train_xs[solver].extend([list(features.tolist())] * 4)
                train_ys[solver].extend([0] * failures + [1] * successes)

        # fit solver models
        self._models = {}

        for solver in solver_index:
            self._models[solver] = model = scikits.learn.linear_model.LogisticRegression()

            model.fit(train_xs[solver], train_ys[solver])

    def __call__(self, cnf_path, budget):
        # obtain features
        csv_path = cnf_path + ".features.csv"

        if os.path.exists(csv_path):
            features = numpy.recfromcsv(csv_path).tolist()
        else:
            (_, features) = borg.features.get_features_for(cnf_path)

        features_cost = features[0]

        # select a solver
        scores = [(s, m.predict_proba([features])[0, -1]) for (s, m) in self._models.items()]
        (name, probability) = max(scores, key = lambda (_, p): p)
        selected = self._solvers[name]

        logger.debug("selected %s on %s", name, os.path.basename(cnf_path))

        # run the solver
        (run_cost, run_answer, _) = selected(cnf_path, budget - features_cost)

        return (features_cost + run_cost, run_answer, None)

named = {
    "random": RandomPortfolio,
    #"baseline": BaselinePortfolio,
    #"oracle": OraclePortfolio,
    #"uber-oracle": UberOraclePortfolio,
    #"classifier": ClassifierPortfolio,
    }

