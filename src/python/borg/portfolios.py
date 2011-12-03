"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import itertools
import numpy
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

class RandomPortfolio(object):
    """Random portfolio."""

    def __call__(self, task, suite, budget):
        """Run the portfolio."""

        solvers = suite.solvers.values()
        selected = numpy.random.randint(len(solvers))

        return solvers[selected].start(task).run_then_stop(budget.cpu_seconds)

class UniformPortfolio(object):
    """Portfolio that runs every solver once."""

    def __call__(self, task, suite, budget):
        """Run the portfolio."""

        budget_each = budget.cpu_seconds / (len(suite.solvers) * 100)
        processes = [s.start(task) for s in suite.solvers.values()]
        next_process = itertools.cycle(processes)

        def finished():
            return \
                budget.cpu_seconds - sum(p.elapsed for p in processes) < budget_each \
                or all(p.terminated for p in processes)

        while not finished():
            process = next_process.next()

            if not process.terminated:
                answer = process.run_then_pause(budget_each)

                if suite.domain.is_final(task, answer):
                    return answer

        return None

class BaselinePortfolio(object):
    """Portfolio that runs the best train-set solver."""

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

        # XXX fix the solver_names situation

        # grab known run data
        budget_count = 100
        solver_names = sorted(suite.solvers)
        data = suite.run_data.filter(task)
        bins = data.to_bins_array(solver_names, budget_count)[0].astype(numpy.double) + 1e-64
        rates = bins / numpy.sum(bins, axis = -1)[..., None]
        log_survival = numpy.log(1.0 - numpy.cumsum(rates[:, :-1], axis = -1))

        # make a plan
        interval = data.get_common_budget() / budget_count
        planner = borg.planners.KnapsackMultiversePlanner()
        #planner = borg.planners.BellmanPlanner()
        plan = planner.plan(log_survival[None, ...])

        # and follow through
        remaining = budget.cpu_seconds

        for (s, b) in plan:
            this_budget = (b + 1) * interval

            assert remaining - this_budget > -1e-1

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

        # XXX again, fix the solver names order mess

        # grab known run data
        budget_count = 10
        self._solver_names = sorted(suite.solvers)
        bins = training.to_bins_array(self._solver_names, budget_count).astype(numpy.double) + 1e-8 / budget_count
        rates = bins / numpy.sum(bins, axis = -1)[..., None]
        survival = 1.0 - numpy.cumsum(rates, axis = -1)

        # make a plan
        self._interval = training.get_common_budget() / budget_count

        planner = borg.planners.KnapsackMultiversePlanner()
        #planner = borg.planners.BellmanPlanner()

        self._plan = planner.plan(numpy.log(survival[..., :-1]))

        logger.info("preplanned plan: %s", self._plan)

    def __call__(self, task, suite, budget):
        """Run the portfolio."""

        remaining = budget.cpu_seconds

        for (s, b) in self._plan:
            this_budget = (b + 1) * self._interval

            assert remaining - this_budget > -1e-1

            process = suite.solvers[self._solver_names[s]].start(task)
            answer = process.run_then_stop(this_budget)
            remaining -= this_budget

            if suite.domain.is_final(task, answer):
                return answer

        return None

class PureModelPortfolio(object):
    """Hybrid mixture-model portfolio."""

    def __init__(self, suite, model, regress = None):
        """Initialize."""

        # XXX fix the solver_names (ordering) situation
        # XXX fix the feature ordering situation

        self._solver_names = sorted(suite.solvers)
        self._solver_name_index = dict(map(reversed, enumerate(self._solver_names)))

        print list(enumerate(self._solver_names))

        self._model = model
        self._planner = borg.planners.KnapsackMultiversePlanner()
        #self._planner = borg.planners.BellmanPlanner()
        self._regress = regress
        self._runs_limit = 256

        self._plan_cache = {}
        self._plan = None

    def __call__(self, task, suite, budget):
        """Run the portfolio."""

        remaining = budget.cpu_seconds

        (feature_names, feature_values) = suite.domain.compute_features(task)
        feature_dict = dict(zip(feature_names, feature_values))
        feature_values_sorted = [feature_dict[f] for f in sorted(feature_names)]
        (predicted_weights,) = numpy.log(self._regress.predict([feature_values_sorted]))

        #if self._plan is None:
        plan = self._planner.plan(self._model.log_survival[..., :-1], predicted_weights)

        #plan = self._plan

        for (s, b) in plan:
            this_budget = (b + 1) * self._model.interval

            assert remaining - this_budget > -1e-1

            process = suite.solvers[self._solver_names[s]].start(task)
            answer = process.run_then_stop(this_budget)
            remaining -= this_budget

            if suite.domain.is_final(task, answer):
                return answer

        return None

    #def __call__(self, task, suite, budget):
        #"""Run the portfolio."""

        #with borg.accounting():
            #return self._solve(task, suite, budget)

    def _solve(self, task, suite, budget):
        """Run the portfolio."""

        # select a solver
        failures = []
        answer = None

        for i in xrange(self._runs_limit):
            # make a plan
            posterior = self._model.condition(failures)
            position = sum(c + 1 for (_, c) in failures)
            plan_key = (position, tuple(sorted(failures)))
            plan = self._plan_cache.get(plan_key)

            if plan is None:
                plan = self._planner.plan(posterior.log_survival[..., :-position - 1], posterior.log_weights)

                self._plan_cache[plan_key] = plan

            if len(plan) == 0:
                break

            #print "plan", plan

            # and follow through
            (plan0_s, plan0_c) = plan[0]
            plan0_budget = self._model.interval * (plan0_c + 1)

            process = suite.solvers[self._solver_names[plan0_s]].start(task)
            answer = process.run_then_stop(plan0_budget)

            if suite.domain.is_final(task, answer):
                return answer

            failures.append((plan0_s, plan0_c))

        return None

