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

class RandomPortfolio(object):
    """Random portfolio."""

    def __call__(self, task, suite, budget):
        """Run the portfolio."""

        return cargo.grab(suite.solvers.values())(task)(budget.cpu_seconds)

class BaselinePortfolio(object):
    """Baseline portfolio."""

    def __init__(self, suite, training):
        """Initialize."""

        solver_names = list(suite.solvers)
        outcome_counts = training.to_bins_array(solver_names, 1).astype(numpy.double)
        success_rates = outcome_counts[..., 0] / numpy.sum(outcome_counts, axis = -1)
        mean_rates = numpy.mean(success_rates, axis = 0)

        self._solver_name = solver_names[numpy.argmax(mean_rates)]

    def __call__(self, task, suite, budget):
        """Run the portfolio."""

        process = suite.solvers[self._solver_name].start(task)
        
        return process.run_then_stop(budget.cpu_seconds)

class OraclePortfolio(object):
    """Optimal prescient discrete-budget portfolio."""

    def __call__(self, task, suite, budget):
        """Run the portfolio."""

        # grab known run data
        budget_count = 100
        solver_names = list(suite.solvers)
        data = suite.runs_data.filter(task)
        bins = data.to_bins_array(solver_names, budget_count)[0].astype(numpy.double) + 1e-64
        rates = bins / numpy.sum(bins, axis = -1)[..., None]
        log_survival = numpy.log(1.0 - numpy.cumsum(rates[:, :-1], axis = -1))

        # make a plan
        interval = data.get_common_budget() / budget_count
        planner = borg.planners.KnapsackMultiversePlanner()
        plan = planner.plan(log_survival[None, ...])

        # and follow through
        remaining = budget.cpu_seconds

        for (s, b) in plan:
            this_budget = (b + 1) * interval

            assert remaining >= this_budget

            process = suite.solvers[solver_names[s]].start(task)
            answer = process.run_then_stop(this_budget)
            remaining -= this_budget

            if suite.domain.is_final(task, answer):
                return answer

        return None

class PreplanningPortfolio(object):
    """Preplanning discrete-budget portfolio."""

    def __init__(self, suite, training):
        """Initialize."""

        # grab known run data
        budget_count = 10
        solver_names = list(suite.solvers)
        bins = training.to_bins_array(solver_names, budget_count).astype(numpy.double) + 1e-64
        rates = bins / numpy.sum(bins, axis = -1)[..., None]
        log_survival = numpy.log(1.0 - numpy.cumsum(rates[..., :-1], axis = -1))

        # make a plan
        self._interval = training.get_common_budget() / budget_count

        #planner = borg.planners.KnapsackMultiversePlanner()
        planner = borg.planners.BellmanPlanner()

        self._plan = planner.plan(log_survival)
        self._solver_names = list(suite.solvers)

        logger.info("preplanned plan: %s", self._plan)

    def __call__(self, task, suite, budget):
        """Run the portfolio."""

        remaining = budget.cpu_seconds

        for (s, b) in self._plan:
            this_budget = (b + 1) * self._interval

            assert remaining >= this_budget

            process = suite.solvers[self._solver_names[s]].start(task)
            answer = process.run_then_stop(this_budget)
            remaining -= this_budget

            if suite.domain.is_final(task, answer):
                return answer

        return None

class PureModelPortfolio(object):
    """Hybrid mixture-model portfolio."""

    def __init__(self, suite, model):
        """Initialize."""

        self._solver_names = list(suite.solvers)
        self._solver_name_index = dict(map(reversed, enumerate(self._solver_names)))

        self._model = model
        self._planner = borg.planners.KnapsackMultiversePlanner()
        #self._planner = borg.planners.BellmanPlanner()
        self._runs_limit = 256

    def __call__(self, task, suite, budget):
        """Run the portfolio."""

        with borg.accounting():
            return self._solve(task, suite, budget)

    def _solve(self, task, suite, budget):
        """Run the portfolio."""

        # select a solver
        failed = []
        answer = None

        for i in xrange(self._runs_limit):
            # obtain model predictions
            failures = []

            for solver in failed:
                failures.append((self._solver_names.index(solver.name), solver.elapsed / self._model.interval))

            posterior = self._model.condition(failures)

            with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
                print "marginal:"
                z = numpy.logaddexp.reduce(posterior.log_survival + posterior.log_weights[..., None, None], axis = 0)
                print numpy.exp(z)

            # make a plan
            position = int(borg.get_accountant().total.cpu_seconds / self._model.interval)
            plan = self._planner.plan(posterior.log_survival[..., position:-1], posterior.log_weights)

            if len(plan) == 0:
                break

            # and follow through
            (plan0_s, plan0_c) = plan[0]
            plan0_budget = self._model.interval * (plan0_c + 1)

            process = suite.solvers[self._solver_names[plan0_s]].start(task)
            answer = process.run_then_stop(plan0_budget)

            if suite.domain.is_final(task, answer):
                return answer

            failed.append(process)

        return None

