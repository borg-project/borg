"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import time
import uuid
import resource
import itertools
import multiprocessing
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def outcome_matrices_from_runs(solver_index, budgets, runs):
    """Build run-outcome matrices from records."""

    S = len(solver_index)
    B = len(budgets)
    N = len(runs)
    successes = numpy.zeros((N, S, B))
    attempts = numpy.zeros((N, S))

    for (n, runs_by_solver) in enumerate(runs.values()):
        for (run_solver, runs) in runs_by_solver.items():
            for (run_budget, run_cost, run_answer) in runs:
                s = solver_index.get(run_solver)

                if s is not None and run_budget >= budgets[-1]:
                    b = numpy.digitize([run_cost], budgets)

                    attempts[n, s] += 1.0

                    if b < B and run_answer:
                        successes[n, s, b] += 1.0

    return (successes, attempts)

def outcome_matrices_from_paths(solver_index, budgets, paths):
    """Build run-outcome matrices from records."""

    # XXX again, belongs in the generic "run storage" module

    S = len(solver_index)
    B = len(budgets)
    N = len(paths)
    successes = numpy.zeros((N, S, B))
    attempts = numpy.zeros((N, S))

    for (n, path) in enumerate(paths):
        (runs,) = borg.portfolios.get_task_run_data([path]).values()

        for (run_solver, _, run_budget, run_cost, run_answer) in runs.tolist():
            s = solver_index.get(run_solver)

            if s is not None and run_budget >= budgets[-1]:
                b = numpy.digitize([run_cost], budgets)

                attempts[n, s] += 1.0

                if b < B and run_answer:
                    successes[n, s, b] += 1.0

    return (successes, attempts)

class MultinomialMixturePortfolio(object):
    """Multinomial-mixture portfolio."""

    def __init__(self, domain, train_paths, budget_interval, budget_count):
        # build action set
        self._domain = domain
        self._budgets = [b * budget_interval for b in xrange(1, budget_count + 1)]

        # acquire running time data
        self._solver_names = list(self._domain.solvers)
        self._solver_name_index = dict(map(reversed, enumerate(self._solver_names)))
        self._budget_index = dict(map(reversed, enumerate(self._budgets)))

        (successes, attempts) = \
            outcome_matrices_from_paths(
                self._solver_name_index,
                self._budgets,
                train_paths,
                )

        logger.info("solvers: %s", dict(enumerate(self._solver_names)))

        # fit our model
        #self._model = borg.models.MultinomialMixtureModel(successes, attempts)
        self._model = borg.models.DCM_MixtureModel(successes, attempts)

    def __call__(self, task, budget, cores = 1):
        if cores != 1:
            raise RuntimeError("no classic (AAAI) support for parallel portfolios")

        with borg.accounting():
            return self._solve(task, budget, cores)

    def _solve(self, task, budget, cores):
        # select a solver
        failures = []

        while True:
            # obtain model predictions
            (class_weights_K, class_rates_KSB) = self._model.predict(failures)
            (K, S, B) = class_rates_KSB.shape
            mean_rates_SB = numpy.sum(class_weights_K[:, None, None] * class_rates_KSB, axis = 0)

            # make a plan...
            remaining = budget - borg.get_accountant().total
            normal_cpu_budget = borg.machine_to_normal(borg.unicore_cpu_budget(remaining))
            (feasible_b,) = numpy.digitize([normal_cpu_budget], self._budgets)

            if feasible_b == 0:
                return None

            discounts_B = (1.0 - 1e-4)**numpy.array(self._budgets)
            discounted_SB = mean_rates_SB * discounts_B
            discounted_Sb = discounted_SB[:feasible_b]
            (s, b) = numpy.unravel_index(numpy.argmax(discounted_Sb), discounted_Sb.shape)
            name = self._solver_names[s]
            solver = self._domain.solvers[name]
            planned_cpu_cost = self._budgets[b]

            # don't waste our final seconds before the buzzer
            if normal_cpu_budget - planned_cpu_cost < self._budgets[0]:
                planned_cpu_cost = normal_cpu_budget
            else:
                planned_cpu_cost = planned_cpu_cost

            machine_cost = borg.normal_to_machine(planned_cpu_cost)

            # be informative
            logger.info(
                "running %s for %i with %i remaining (marginal = %.2f)" % (
                    name,
                    machine_cost,
                    remaining.cpu_seconds,
                    mean_rates_SB[s, b],
                    ),
                )

            answer = solver(task)(machine_cost)

            if self._domain.is_final(task, answer):
                logger.info("run succeeded!")

                return answer
            else:
                failures.append((s, b))

borg.portfolios.named["aaai-mul"] = MultinomialMixturePortfolio

