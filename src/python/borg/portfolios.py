"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import resource
import itertools
import contextlib
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

        for (name, cost) in plan:
            solver = self._solvers[name]
            (run_cost, answer) = solver(cnf_path, cost)

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

    concentration = 1e-8
    N = observed.shape[0]
    rates = (observed + concentration / 2.0) / (counts + concentration)
    components = rates[numpy.random.randint(N, size = K)]
    responsibilities = numpy.empty((K, N))
    old_ll = -numpy.inf

    for i in xrange(512):
        # compute new responsibilities
        raw_mass = scipy.stats.binom.logpmf(observed[None, ...], counts[None, ...], components[:, None, ...])
        log_mass = numpy.sum(numpy.sum(raw_mass, axis = 3), axis = 2)
        responsibilities = log_mass - numpy.logaddexp.reduce(log_mass, axis = 0)
        responsibilities = numpy.exp(responsibilities)
        weights = numpy.sum(responsibilities, axis = 1)
        weights /= numpy.sum(weights)
        ll = numpy.sum(numpy.logaddexp.reduce(numpy.log(weights)[:, None] + log_mass, axis = 0))

        logger.info("l-l at iteration %i is %f", i, ll)

        # compute new components
        map_observed = observed[None, ...] * responsibilities[..., None, None]
        map_counts = counts[None, ...] * responsibilities[..., None, None]
        components = numpy.sum(map_observed, axis = 1) + concentration / 2.0
        components /= numpy.sum(map_counts, axis = 1) + concentration

        # check for termination
        if numpy.abs(ll - old_ll) <= 1e-3:
            break

        old_ll = ll

    assert numpy.all(components >= 0.0)
    assert numpy.all(components <= 1.0)

    return (components, weights, responsibilities, ll)

class MixturePortfolio(object):
    """Mixture-model portfolio."""

    def __init__(self, solvers, train_paths):
        # build action set
        self._solvers = solvers
        self._budgets = budgets = [(b + 1) * 100.0 for b in xrange(50)]

        # acquire running time data
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

        # fit the mixture model
        best = -numpy.inf

        for K in [128]:
            (components, weights, responsibilities, ll) = fit_mixture_model(observed, counts, K)

            logger.info("model with K = %i has l-l = %f", K, ll)

            if ll > best:
                self._components = components
                self._weights = weights
                best = ll

        logger.info("fit the mixture model; best l-l = %f", best)

        # fit the cluster classifiers
        train_xs = map(lambda _: [], xrange(K))
        train_ys = map(lambda _: [], xrange(K))

        for (i, train_path) in enumerate(train_paths):
            features = numpy.recfromcsv(train_path + ".features.csv").tolist()

            for k in xrange(K):
                rows = 100
                positive = int(round(responsibilities[k, i] * rows))
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

        logger.info("trained the cluster classifiers")

    def __call__(self, cnf_path, budget):
        # obtain features
        csv_path = cnf_path + ".features.csv"

        if os.path.exists(csv_path):
            features = numpy.recfromcsv(csv_path).tolist()
        else:
            (_, features) = borg.features.get_features_for(cnf_path)

        features_cost = features[0]

        # use classifiers to establish initial cluster probabilities
        #prior_weights = numpy.array([m.predict_proba([features])[0, -1] for m in self._models])
        #prior_weights /= numpy.sum(prior_weights)

        # use oracle knowledge to establish initial cluster probabilities
        (runs,) = get_task_run_data([cnf_path]).values()
        (oracle_history, oracle_counts, _) = \
            action_rates_from_runs(
                self._solver_index,
                self._budget_index,
                runs.tolist(),
                )
        true_rates = oracle_history / oracle_counts
        raw_mass = scipy.stats.binom.logpmf(oracle_history, oracle_counts, self._components)
        log_mass = numpy.sum(numpy.sum(raw_mass, axis = 2), axis = 1)
        prior_weights = numpy.exp(log_mass - numpy.logaddexp.reduce(log_mass))
        first_rates = numpy.sum(self._components * prior_weights[:, None, None], axis = 0)

        # select a solver
        initial_utime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        total_run_cost = features_cost
        history = numpy.zeros((len(self._solvers), len(self._budgets)))
        new_weights = prior_weights
        answer = None

        logger.info("solving %s", cnf_path)

        while True:
            # compute marginal probabilities of success
            #overhead = resource.getrusage(resource.RUSAGE_SELF).ru_utime - initial_utime
            overhead = 0.0 # XXX
            total_cost = total_run_cost + overhead

            with fpe_handling(under = "ignore"):
                rates = numpy.sum(self._components * new_weights[:, None, None], axis = 0)

            # generate a plan
            plan = knapsack_plan(rates, self._solver_rindex, self._budgets, budget - total_cost)

            if not plan:
                break

            # XXX if our belief is zero, bail or otherwise switch to plan B

            (name, planned_cost) = plan[0]

            if budget - total_cost - planned_cost < self._budgets[0]:
                max_cost = budget - total_cost
            else:
                max_cost = planned_cost

            # run it
            logger.info(
                "taking %s@%i with %i remaining (b = %.2f; p = %.2f)",
                name,
                max_cost,
                budget - total_cost,
                rates[self._solver_index[name], self._budget_index[planned_cost]],
                true_rates[self._solver_index[name], self._budget_index[planned_cost]],
                )

            # XXX adjust cost
            solver = self._solvers[name]
            (run_cost, answer) = solver(cnf_path, max_cost)
            total_run_cost += run_cost

            if answer is not None:
                break

            # recalculate cluster probabilities
            history[self._solver_index[name]] += 1.0
            # XXX uncomment
            #new_weights = scipy.stats.binom.pmf(numpy.zeros_like(history), history, self._components)
            #new_weights = numpy.prod(numpy.prod(new_weights, axis = 2), axis = 1)
            #new_weights *= prior_weights
            #new_weights /= numpy.sum(new_weights)

            raw_mass = scipy.stats.binom.logpmf(oracle_history, oracle_counts + history, self._components)
            log_mass = numpy.sum(numpy.sum(raw_mass, axis = 2), axis = 1)
            new_weights = numpy.exp(log_mass - numpy.logaddexp.reduce(log_mass))

        total_cost = total_run_cost + overhead

        logger.info("answer %s with total cost %.2f", answer, total_cost)

        if answer is None:
            print self._solver_rindex
            print "truth:"
            print "\n".join(" ".join(map("{0:02.0f}".format, row * 100)) for row in true_rates)
            print "initial beliefs:"
            print "\n".join(" ".join(map("{0:02.0f}".format, row * 100)) for row in first_rates)
            print "final beliefs:"
            print "\n".join(" ".join(map("{0:02.0f}".format, row * 100)) for row in rates)

        return (total_cost, answer)

named = {
    #"random": RandomPortfolio,
    #"oracle": OraclePortfolio,
    #"uber-oracle": UberOraclePortfolio,
    #"baseline": BaselinePortfolio,
    #"classifier": ClassifierPortfolio,
    "mixture": MixturePortfolio,
    }

