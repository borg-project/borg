"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import resource
import itertools
import numpy
import scipy.stats
import scikits.learn.linear_model
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

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

        (runs,) = get_task_run_data([cnf_path]).values()
        (_, _, rates) = action_rates_from_runs(self._solver_index, self._budget_index, runs.tolist())

        values = numpy.ones(len(self._budgets) + 1)
        policy = {}

        for b in xrange(1, len(self._budgets) + 1):
            region = (1.0 - rates[:, :b]) * values[None, b - 1::-1]
            (s, c) = numpy.unravel_index(numpy.argmin(region), region.shape)
            values[b] = region[s, c]
            action = (
                self._solver_rindex[s],
                self._budget_rindex[c],
                )
            policy[self._budgets[b - 1]] = action

        plan = []
        remaining = budget

        while remaining > 0.0:
            (_, cost) = action = policy[remaining]

            plan.append(action)

            remaining -= cost

        def heuristic((name, cost)):
            return -rates[self._solver_index[name], self._budget_index[cost]] / cost

        plan = sorted(plan, key = heuristic)

        total_cost = 0.0

        for (name, cost) in plan:
            solver = self._solvers[name]
            (run_cost, answer) = solver(cnf_path, cost)

            b = self._budget_index[cost]
            logger.debug(
                "ran %s@%i with %is; yielded %s (p = %f; v = %f)",
                name,
                cost,
                budget - total_cost,
                answer,
                rates[self._solver_index[name], b],
                values[b + 1],
                )

            total_cost += run_cost

            if answer is not None:
                return (total_cost, answer)

        return (total_cost, None)

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
        (run_cost, run_answer) = selected(cnf_path, budget - features_cost)

        return (features_cost + run_cost, run_answer)

def fit_mixture_model(observed, counts, K):
    """Use EM to fit a discrete mixture."""

    N = observed.shape[0]
    components = numpy.empty((K,) + observed.shape[1:])
    responsibilities = numpy.empty((K, N))
    concentration = 1e-6

    for k in xrange(K):
        n = numpy.random.randint(N)
        components[k] = (observed[n] + concentration / 2.0) / (counts[n] + concentration)

    for i in xrange(64):
        # compute new responsibilities
        for k in xrange(K):
            mass = scipy.stats.binom.pmf(observed, counts, components[k])
            responsibilities[k] = numpy.prod(numpy.prod(mass, axis = 2), axis = 1)

        responsibilities /= numpy.sum(responsibilities, axis = 0)

        # compute new components
        for k in xrange(K):
            components[k] = numpy.sum(observed * responsibilities[k][:, None, None], axis = 0)
            components[k] += concentration / 2.0
            components[k] /= numpy.sum(counts * responsibilities[k][:, None, None], axis = 0) + concentration

    weights = numpy.sum(responsibilities, axis = 1)
    weights /= numpy.sum(weights)

    assert numpy.all(components >= 0.0)
    assert numpy.all(components <= 1.0)

    return (components, weights, responsibilities)

class MixturePortfolio(object):
    """Mixture-model portfolio."""

    def __init__(self, solvers, train_paths):
        self._solvers = solvers
        #self._budgets = budgets = numpy.r_[10:4000:12j]
        self._budgets = budgets = [5000]

        # fit the mixture model
        K = 32
        self._solver_rindex = dict(enumerate(solvers))
        self._solver_index = dict(map(reversed, self._solver_rindex.items()))
        budget_rindex = dict(enumerate(budgets))
        self._budget_index = dict(map(reversed, budget_rindex.items()))
        observed = numpy.empty((len(train_paths), len(solvers), len(budgets)))
        counts = numpy.empty((len(train_paths), len(solvers), len(budgets)))

        for (i, train_path) in enumerate(train_paths):
            (runs,) = get_task_run_data([train_path]).values()
            (runs_observed, runs_counts, _) = \
                action_rates_from_runs(
                    self._solver_index,
                    self._budget_index,
                    runs.tolist(),
                    )

            observed[i] = runs_observed
            counts[i] = runs_counts

        best = -numpy.inf

        for i in xrange(8):
            (components, weights, _) = fit_mixture_model(observed, counts, K)
            ps = scipy.stats.binom.pmf(observed[None, ...], counts[None, ...], components[:, None, ...])
            p = numpy.sum(numpy.prod(ps * weights[:, None, None, None], axis = 1))

            if p > best:
                self._components = components
                self._weights = weights
                best = p

        # fit the cluster classifiers
        targets = numpy.empty((K, observed.shape[0]))

        for k in xrange(K):
            mass = scipy.stats.binom.pmf(observed, counts, self._components[k])
            targets[k] = numpy.prod(numpy.prod(mass, axis = 2), axis = 1)

        train_xs = map(lambda _: [], xrange(K))
        train_ys = map(lambda _: [], xrange(K))

        for (i, train_path) in enumerate(train_paths):
            features = numpy.recfromcsv(train_path + ".features.csv").tolist()

            for k in xrange(K):
                rows = 100
                positive = int(round(targets[k, i] * rows))
                negative = rows - positive

                train_xs[k].extend([features] * positive)
                train_ys[k].extend([1] * positive)

                train_xs[k].extend([features] * negative)
                train_ys[k].extend([0] * negative)

        self._models = []

        for k in xrange(K):
            model = scikits.learn.linear_model.LogisticRegression()

            model.fit(train_xs[k], train_ys[k])

            self._models.append(model)

    def __call__(self, cnf_path, budget):
        ## obtain features
        csv_path = cnf_path + ".features.csv"

        if os.path.exists(csv_path):
            features = numpy.recfromcsv(csv_path).tolist()
        else:
            (_, features) = borg.features.get_features_for(cnf_path)

        prior_weights = numpy.array([m.predict_proba([features])[0, -1] for m in self._models])
        prior_weights /= numpy.sum(prior_weights)

        #features_cost = features[0]

        #print new_weights

        #(runs,) = get_task_run_data([cnf_path]).values()
        #(oracle_history, oracle_counts, _) = \
            #action_rates_from_runs(
                #self._solver_index,
                #self._budget_index,
                #runs.tolist(),
                #)
        #new_weights = scipy.stats.binom.pmf(oracle_history, oracle_counts, self._components)
        #new_weights = numpy.prod(numpy.prod(new_weights, axis = 2), axis = 1)
        #new_weights /= numpy.sum(new_weights)

        #print new_weights

        # select a solver
        features_cost = 0.0
        #prior_weights = self._weights
        initial_utime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        total_cost = 0.0
        total_run_cost = features_cost
        history = numpy.zeros((len(self._solvers), len(self._budgets)))
        new_weights = prior_weights

        #logger.debug("solving %s", cnf_path)

        while True:
            # choose an action
            answer = None
            overhead = resource.getrusage(resource.RUSAGE_SELF).ru_utime - initial_utime
            total_cost = total_run_cost + overhead
            probabilities = numpy.sum(self._components * new_weights[:, None, None], axis = 0)
            best_score = -numpy.inf

            for cost in sorted(self._budgets):
                adjusted_cost = borg.defaults.machine_speed * cost

                if adjusted_cost <= budget - total_cost:
                    #discount = (1.0 - 5e-4)**cost
                    discount = 1.0
                    budget_i = self._budget_index[cost]
                    solver_i = numpy.argmax(probabilities[:, budget_i] * discount)
                    score = probabilities[solver_i, budget_i] * discount

                    if score >= best_score:
                        best_score = score
                        best_name = self._solver_rindex[solver_i]
                        best = self._solvers[best_name]
                        best_budget = adjusted_cost
                        best_solver_i = solver_i
                        best_budget_i = budget_i

            if best_score < 0.0:
                break

            logger.info("action %s@%i was chosen with score %.2f", best_name, best_budget, best_score)

            # run it
            (run_cost, answer) = best(cnf_path, best_budget)
            total_run_cost += run_cost

            if answer is not None:
                break

            # recalculate cluster probabilities
            history[best_solver_i] += 1.0
            new_weights = scipy.stats.binom.pmf(numpy.zeros_like(history), history, self._components)
            new_weights = numpy.prod(numpy.prod(new_weights, axis = 2), axis = 1)
            new_weights *= prior_weights
            new_weights /= numpy.sum(new_weights)

        total_cost = total_run_cost + overhead

        logger.debug("answer %s with total cost %.2f", answer, total_cost)

        return (total_cost, answer)

named = {
    "random": RandomPortfolio,
    "oracle": OraclePortfolio,
    "uber-oracle": UberOraclePortfolio,
    "baseline": BaselinePortfolio,
    "classifier": ClassifierPortfolio,
    "mixture": MixturePortfolio,
    }

