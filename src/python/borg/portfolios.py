"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import uuid
import contextlib
import multiprocessing
import numpy
import scikits.learn.linear_model
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@contextlib.contextmanager
def fpe_handling(**kwargs):
    previous = numpy.seterr(**kwargs)

    yield

    numpy.seterr(**previous)

# XXX this code should be moved into a "run outcome storage" module

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

                    if run_cost <= budget and run_answer:
                        observed[solver_i, budget_i] += 1.0

    return (observed, counts, observed / counts)

class RandomPortfolio(object):
    """Random-action portfolio."""

    def __init__(self, domain, train_paths, budget_interval, budget_count):
        self._domain = domain
    
    def __call__(self, task, budget, cores = 1):
        queue = multiprocessing.Queue()
        solvers = []

        for _ in xrange(cores):
            solver_id = uuid.uuid4()
            solver = cargo.grab(self._domain.solvers.values())(task, queue, solver_id)

            solvers.append(solver)

        try:
            for solver in solvers:
                solver.unpause_for(borg.unicore_cpu_budget(budget))

            remaining = len(solvers)

            while remaining > 0:
                (solver_id, run_cpu_cost, answer, _) = queue.get()

                remaining -= 1

                borg.get_accountant().charge_cpu(run_cpu_cost)

                if self._domain.is_final(task, answer):
                    return answer

            return None
        finally:
            for solver in solvers:
                solver.stop()

class BaselinePortfolio(object):
    """Baseline portfolio."""

    def __init__(self, domain, train_paths, budget_interval, budget_count):
        solver_rindex = dict(enumerate(domain.solvers))
        solver_index = dict(map(reversed, solver_rindex.items()))
        train = get_task_run_data(train_paths)
        rates = None

        for runs in train.values():
            (_, _, task_rates) = \
                action_rates_from_runs(
                    solver_index,
                    {budget_interval * budget_count: 0},
                    runs.tolist(),
                    )

            if rates is None:
                rates = task_rates
            else:
                rates += task_rates

        rates /= len(train)

        self._best = domain.solvers[solver_rindex[numpy.argmax(rates)]]

    def __call__(self, task, budget, cores = 1):
        assert cores == 1

        return self._best(task)(borg.unicore_cpu_budget(budget))

class OraclePortfolio(object):
    """Optimal discrete-budget portfolio."""

    def __init__(self, bundle, training, budget_interval, budget_count):
        """Initialize."""

        self._budgets = [b * budget_interval for b in xrange(1, budget_count + 1)]
        self._solver_rindex = dict(enumerate(bundle.solvers))
        self._solver_index = dict(map(reversed, self._solver_rindex.items()))
        self._budget_rindex = dict(enumerate(self._budgets))

    def __call__(self, task, bundle, budget, cores = 1):
        """Run the portfolio."""

        assert cores == 1
        assert bundle.domain.name == "fake"

        # grab known run data
        runs = bundle.runs_data.get_run_list(task)

        (successes, attempts) = \
            borg.storage.outcome_matrices_from_runs(
                self._solver_index,
                self._budgets,
                {None: runs},
                )
        rates = (successes / attempts[..., None])[0, ...]

        # make a plan
        plan = \
            borg.bilevel.plan_knapsack_multiverse(
                numpy.array([1.0]),
                numpy.array([rates]),
                )

        # and follow through
        for (solver_i, budget_i) in plan:
            solver = self._solver_rindex[solver_i]
            budget = self._budget_rindex[budget_i]
            answer = bundle.solvers[solver](task)(budget)

            if bundle.domain.is_final(task, answer):
                return answer

        return None

class PreplanningPortfolio(object):
    """Optimal discrete-budget portfolio."""

    def __init__(self, bundle, training, budget_interval, budget_count):
        """Initialize."""

        self._budgets = [b * budget_interval for b in xrange(1, budget_count + 1)]
        self._solver_rindex = dict(enumerate(bundle.solvers))
        self._solver_index = dict(map(reversed, self._solver_rindex.items()))
        self._budget_rindex = dict(enumerate(self._budgets))

        # make a plan
        run_lists = training.get_run_lists()

        (successes, attempts) = \
            borg.storage.outcome_matrices_from_runs(
                self._solver_index,
                self._budgets,
                run_lists,
                )

        self._plan = \
            borg.bilevel.plan_knapsack_multiverse(
                numpy.ones(len(run_lists)) / len(run_lists),
                numpy.array(successes / attempts[..., None]),
                )

    def __call__(self, task, bundle, budget, cores = 1):
        """Run the portfolio."""

        assert cores == 1

        for (solver_i, budget_i) in self._plan:
            solver = self._solver_rindex[solver_i]
            budget = self._budget_rindex[budget_i]
            answer = bundle.solvers[solver](task)(budget)

            if bundle.domain.is_final(task, answer):
                return answer

        return None

class ClassifierPortfolio(object):
    """Classifier-based portfolio."""

    def __init__(self, bundle, training, budget_interval, budget_count):
        """Initialize."""

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

    def __call__(self, task, budget, cores = 1):
        """Run the portfolio."""

        assert cores == 1

        # obtain features
        with borg.accounting() as features_accountant:
            (_, features) = self._domain.compute_features(task)

        # select a solver
        scores = [(s, m.predict_proba([features])[0, -1]) for (s, m) in self._models.items()]
        (name, probability) = max(scores, key = lambda (_, p): p)

        # run the solver
        run_cpu_budget = borg.unicore_cpu_budget(budget - features_accountant.total)

        return self._domain.solvers[name](task)(run_cpu_budget)

class MixturePortfolio(object):
    """Hybrid mixture-model portfolio."""

    RUNS_LIMIT = 256
    BUDGET_INTERVAL = 100

    def __init__(self, bundle, model):
        """Initialize."""

        self._solver_names = list(bundle.solvers)
        self._solver_name_index = dict(map(reversed, enumerate(self._solver_names)))

        self._model = model
        self._planner = \
            borg.planners.KnapsackMultiversePlanner(
                self._solver_name_index,
                self.BUDGET_INTERVAL,
                )

        logger.info("solvers: %s", dict(enumerate(self._solver_names)))

    def __call__(self, task, bundle, budget, cores = 1):
        """Run the portfolio."""

        with borg.accounting():
            return self._solve(task, bundle, budget, cores)

    def _solve(self, task, bundle, budget, cores):
        """Run the portfolio."""

        # select a solver
        S = len(self._solver_names)
        queue = multiprocessing.Queue()
        running = {}
        paused = []
        failed = []
        answer = None

        for i in xrange(self.RUNS_LIMIT):
            # obtain model predictions
            failures = []

            for solver in failed + paused + running.values():
                failures.append((solver.s, solver.cpu_budgeted))

            posterior = self._model.condition(failures)

            # make a plan...
            remaining = budget - borg.get_accountant().total
            normal_cpu_budget = borg.machine_to_normal(borg.unicore_cpu_budget(remaining))

            if normal_cpu_budget < 0.0:
                break

            raw_plan = self._planner.plan(posterior, normal_cpu_budget)

            if len(raw_plan) == 0:
                break

            # interpret the plan's first action
            (a, planned_cpu_cost) = raw_plan[0]

            if a >= S:
                solver = paused.pop(a - S)
                s = solver.s
                name = self._solver_names[s]
            else:
                s = a
                name = self._solver_names[s]
                solver = bundle.solvers[name](task, queue, uuid.uuid4())
                solver.s = s
                solver.cpu_cost = 0.0

            # be informative
            logger.info(
                "[run %i] %s@%i for %i with %i remaining" % (
                    i,
                    name,
                    borg.normal_to_machine(solver.cpu_cost),
                    borg.normal_to_machine(planned_cpu_cost),
                    remaining.cpu_seconds,
                    ),
                )

            # ... and follow through
            solver.unpause_for(borg.normal_to_machine(planned_cpu_cost))

            running[solver._solver_id] = solver

            solver.cpu_budgeted = solver.cpu_cost + planned_cpu_cost

            if len(running) == cores:
                response = queue.get()

                if isinstance(response, Exception):
                    raise response
                else:
                    (solver_id, run_cpu_seconds, answer, terminated) = response

                borg.get_accountant().charge_cpu(run_cpu_seconds)

                solver = running.pop(solver_id)

                solver.cpu_cost += borg.machine_to_normal(run_cpu_seconds)

                if bundle.domain.is_final(task, answer):
                    break
                elif terminated:
                    failed.append(solver)
                else:
                    paused.append(solver)

        for process in paused + running.values():
            process.stop()

        return answer

named = {
    #"random": RandomPortfolio,
    #"baseline": BaselinePortfolio,
    "oracle": OraclePortfolio,
    #"classifier": ClassifierPortfolio,
    "preplanning": PreplanningPortfolio,
    "mixture": MixturePortfolio,
    }

