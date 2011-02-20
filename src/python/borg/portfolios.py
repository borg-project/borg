"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import itertools
import numpy
import scipy.stats
import scikits.learn.linear_model
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def get_task_run_data(task_paths):
    """Load running times associated with tasks."""

    logger.detail("loading running time data from %i files", len(task_paths))

    data = {}

    for path in task_paths:
        csv_path = "{0}.rtd.csv".format(path)
        data[path] = numpy.recfromcsv(csv_path)

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
    
    def __call__(self, cnf_path, budget):
        return cargo.grab(self._solvers.values())(cnf_path, budget)

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
            (_, _, task_rates) = action_rates_from_runs(solver_index, budget_index, runs)

            if rates is None:
                rates = task_rates
            else:
                rates += task_rates

        rates /= len(train)

        self._best = solvers[solver_rindex[numpy.argmax(rates)]]

    def __call__(self, cnf_path, budget):
        return self._best(cnf_path, budget)

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
            (_, _, rates) = action_rates_from_runs(solver_index, budget_index, runs)

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
            features_cost = features.cost
        else:
            (_, features) = borg.features.get_features_for(cnf_path)
            features_cost = features[0]

        # select a solver
        scores = [(s, m.predict_proba([features])[0, -1]) for (s, m) in self._models.items()]
        (name, probability) = max(scores, key = lambda (_, p): p)
        selected = self._solvers[name]

        # run the solver
        (run_cost, run_answer) = selected(cnf_path, budget - features_cost)

        return (features_cost + run_cost, run_answer)

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

class MixturePortfolio(object):
    """Mixture-model portfolio."""

    def __init__(self, solvers, train_paths):
        self._solvers = solvers

        self._budgets = budgets = [1000]
        self._solver_rindex = solver_rindex = dict(enumerate(solvers))
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

        (self._components, self._weights) = fit_mixture_model(observed, counts)

    def __call__(self, cnf_path, budget):
        actions = list(itertools.product(self._solvers, self._budgets))
        total_cost = 0.0
        history = numpy.zeros(len(actions))
        new_weights = self._weights

        while actions:
            # choose a solver
            probabilities = numpy.sum(self._components * new_weights[:, None], axis = 0)

            best_solver_i = numpy.argmax(probabilities)
            best_solver_name = self._solver_rindex[best_solver_i]
            best = self._solvers[best_solver_name]

            #print best_solver_name, probabilities, budget - total_cost

            # run it
            (run_cost, run_answer) = best(cnf_path, self._budgets[0]) # XXX
            total_cost += run_cost

            if run_answer is not None:
                return (total_cost, run_answer)

            history[best_solver_i] += 1.0
            new_weights = scipy.stats.binom.pmf(numpy.zeros(len(actions)), history, components)
            new_weights = numpy.prod(new_weights, axis = 1)
            new_weights *= self._weights
            new_weights /= numpy.sum(new_weights)

            actions = filter(lambda (_, c): c <= budget - total_cost, actions)

            # XXX obviously broken in several ways

        return (total_cost, None)

named = {
    #"TNM": lambda *_: lambda *args: solve_fake("TNM", *args),
    #"SATzilla2009_R": lambda *_: lambda *args: solve_fake("SATzilla2009_R", *args),
    "random": RandomPortfolio,
    "baseline": BaselinePortfolio,
    "classifier": ClassifierPortfolio,
    "mixture": MixturePortfolio,
    }

