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

def action_rates_from_runs(domain, solver_index, budget_index, runs):
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

                    if run_cost <= budget and domain.is_final(run_answer):
                        observed[solver_i, budget_i] += 1.0

    return (observed, counts, observed / counts)

class RandomPortfolio(object):
    """Random-action portfolio."""

    def __init__(self, domain, train_paths, budget_interval, budget_count):
        self._domain = domain
    
    def __call__(self, cnf_path, budget, cores = 1):
        queue = multiprocessing.Queue()
        solvers = []

        for _ in xrange(cores):
            solver_id = uuid.uuid4()
            solver = cargo.grab(self._domain.solvers.values())(cnf_path, queue, solver_id)

            solvers.append(solver)

        try:
            for solver in solvers:
                solver.unpause_for(borg.unicore_cpu_budget(budget))

            remaining = len(solvers)

            while remaining > 0:
                (solver_id, run_cpu_cost, answer, _) = queue.get()

                remaining -= 1

                borg.get_accountant().charge_cpu(run_cpu_cost)

                if self._domain.is_final(answer):
                    return answer

            return None
        finally:
            for solver in solvers:
                solver.stop()

class BaselinePortfolio(object):
    """Baseline portfolio."""

    def __init__(self, domain, train_paths, budget_interval, budget_count):
        budgets = [budget_interval * budget_count]
        solver_rindex = dict(enumerate(domain.solvers))
        solver_index = dict(map(reversed, solver_rindex.items()))
        budget_rindex = dict(enumerate(budgets))
        budget_index = dict(map(reversed, budget_rindex.items()))
        train = get_task_run_data(train_paths)
        rates = None

        for runs in train.values():
            (_, _, task_rates) = \
                action_rates_from_runs(
                    domain,
                    solver_index,
                    budget_index,
                    runs.tolist(),
                    )

            if rates is None:
                rates = task_rates
            else:
                rates += task_rates

        rates /= len(train)

        self._best = domain.solvers[solver_rindex[numpy.argmax(rates)]]

    def __call__(self, cnf_path, budget, cores = 1):
        assert cores == 1

        return self._best(cnf_path)(borg.unicore_cpu_budget(budget))

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
    """Optimal discrete-budget portfolio."""

    def __init__(self, domain, train_paths, budget_interval, budget_count):
        self._domain = domain
        self._budgets = [b * budget_interval for b in xrange(1, budget_count + 1)]
        self._solver_rindex = dict(enumerate(domain.solvers))
        self._solver_index = dict(map(reversed, self._solver_rindex.items()))
        self._budget_rindex = dict(enumerate(self._budgets))
        self._budget_index = dict(map(reversed, self._budget_rindex.items()))

    def __call__(self, cnf_path, budget, cores = 1):
        assert cores == 1

        # grab known run data
        (runs,) = get_task_run_data([cnf_path]).values()
        (_, _, rates) = \
            action_rates_from_runs(
                self._domain,
                self._solver_index,
                self._budget_index,
                runs.tolist(),
                )

        # make a plan, and follow through
        plan = \
            knapsack_plan(
                rates,
                self._solver_rindex,
                self._budgets,
                borg.unicore_cpu_budget(budget),
                )

        for (name, cost) in plan:
            answer = self._domain.solvers[name](cnf_path)(cost)

            if self._domain.is_final(answer):
                return answer

        return None

class ClassifierPortfolio(object):
    """Classifier-based portfolio."""

    def __init__(self, domain, train_paths, budget_interval, budget_count):
        self._domain = domain
        action_budget = budget_interval * budget_count
        solver_rindex = dict(enumerate(domain.solvers))
        solver_index = dict(map(reversed, solver_rindex.items()))
        budget_rindex = dict(enumerate([action_budget]))
        budget_index = dict(map(reversed, budget_rindex.items()))

        # prepare training data
        train_xs = dict((s, []) for s in solver_index)
        train_ys = dict((s, []) for s in solver_index)

        for train_path in train_paths:
            features = numpy.recfromcsv(train_path + ".features.csv")
            (runs,) = get_task_run_data([train_path]).values()
            (_, _, rates) = \
                action_rates_from_runs(
                    domain,
                    solver_index,
                    budget_index,
                    runs.tolist(),
                    )

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

    def __call__(self, task_path, budget, cores = 1):
        assert cores == 1

        # obtain features
        with borg.accounting() as features_accountant:
            features = borg.get_features_for(self._domain, task_path)

        # select a solver
        scores = [(s, m.predict_proba([features])[0, -1]) for (s, m) in self._models.items()]
        (name, probability) = max(scores, key = lambda (_, p): p)

        logger.debug("selected %s on %s", name, os.path.basename(task_path))

        # run the solver
        run_cpu_budget = borg.unicore_cpu_budget(budget - features_accountant.total)

        return self._domain.solvers[name](task_path)(run_cpu_budget)

named = {
    "random": RandomPortfolio,
    "baseline": BaselinePortfolio,
    "uber-oracle": UberOraclePortfolio,
    "classifier": ClassifierPortfolio,
    }

