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

def plan_knapsack_multiverse(tclass_weights_W, tclass_rates_WSB):
    """
    Generate an optimal plan for a static set of possible probability tables.

    Budgets *must* be in ascending order and linearly spaced from zero.
    """

    # heuristically filter low-probability actions
    mean_rates_SB = numpy.sum(tclass_weights_W[:, None, None] * tclass_rates_WSB, axis = 0)
    mean_cmf_SB = numpy.cumsum(mean_rates_SB, axis = -1)
    #threshold = 0.10

    #if numpy.all(mean_cmf_SB <= threshold):
        #logger.warning("not filtering tclasses; no entries in mean exceed %.2f", threshold)
    #else:
        #tclass_rates_WSB[:, mean_cmf_SB[:, :1] <= threshold] = 1e-8
        #tclass_rates_WSB[:, mean_cmf_SB <= threshold] = 1e-8

    # convert model predictions appropriately
    log_weights_W = numpy.log(tclass_weights_W)
    log_fail_cmf_WSB = numpy.log(1.0 - numpy.cumsum(tclass_rates_WSB, axis = -1))

    # generate the value table and associated policy
    (W, S, B) = log_fail_cmf_WSB.shape
    values_WB1 = numpy.zeros((W, B + 1))
    policy = {}

    for b in xrange(1, B + 1):
        region_WSb = log_fail_cmf_WSB[..., :b] + values_WB1[:, None, b - 1::-1]
        weighted_Sb = numpy.logaddexp.reduce(region_WSb + log_weights_W[:, None, None], axis = 0)
        i = numpy.argmin(weighted_Sb)
        (s, c) = numpy.unravel_index(i, weighted_Sb.shape)

        values_WB1[:, b] = region_WSb[:, s, c]
        policy[b] = (s, c)

    # build a plan from the policy
    plan = []
    b = B

    while b > 0:
        (_, c) = action = policy[b]
        b -= c + 1

        plan.append(action)

    # heuristically reorder the plan
    def heuristic((s, c)):
        return mean_cmf_SB[s, c] / (c + 1)

    plan = sorted(plan, key = heuristic, reverse = True)

    # ...
    return plan

class BilevelPortfolio(object):
    """Bilevel mixture-model portfolio."""

    def __init__(self, solvers, train_paths):
        # build action set
        self._solvers = solvers
        self._budgets = budgets = [(b + 1) * 50.0 for b in xrange(42)] # XXX

        # acquire running time data
        self._solver_names = list(solvers)
        self._solver_name_index = dict(map(reversed, enumerate(self._solver_names)))
        self._budget_index = dict(map(reversed, enumerate(self._budgets)))

        (successes, attempts) = borg.models.outcome_matrices_from_paths(self._solver_name_index, self._budgets, train_paths)

        logger.info("solvers: %s", dict(enumerate(self._solver_names)))

        # acquire features
        features = [numpy.recfromcsv(path + ".features.csv").tolist() for path in train_paths]

        # fit our model
        self._model = borg.models.BilevelMultinomialModel(successes, attempts, features)

    def __call__(self, cnf_path, budget, cores = 1):
        with borg.accounting():
            return self._solve(cnf_path, budget, cores)

    def _solve(self, cnf_path, budget, cores):
        # obtain features
        features = borg.features.get_features_for(cnf_path)

        # select a solver
        queue = multiprocessing.Queue()
        running = {}
        paused = []
        failed = []
        answer = None

        while True:
            # obtain model predictions
            failed_indices = []

            for solver in failed + paused + running.values():
                (total_b,) = numpy.digitize([solver.cpu_budgeted], self._budgets)
                if total_b == 0: # XXX hack
                    total_b += 1

                failed_indices.append((solver.s, total_b - 1))

            (tclass_weights_L, tclass_rates_LSB) = self._model.predict(failed_indices, features)
            (L, S, B) = tclass_rates_LSB.shape

            # prepare augmented PMF matrix
            augmented_tclass_arrays = [tclass_rates_LSB]

            for solver in paused:
                s = solver.s
                (c,) = numpy.digitize([solver.cpu_cost], self._budgets)
                if c > 0: # XXX hack
                    c -= 1

                paused_tclass_LB = numpy.zeros((L, B))

                paused_tclass_LB[:, :B - c - 1] = tclass_rates_LSB[:, s, c + 1:]
                paused_tclass_LB /= 1.0 - numpy.sum(tclass_rates_LSB[:, s, :c + 1], axis = -1)[..., None]

                augmented_tclass_arrays.append(paused_tclass_LB[:, None, :])

            augmented_tclass_rates_LAB = numpy.hstack(augmented_tclass_arrays)

            # make a plan...
            remaining = budget - borg.get_accountant().total
            normal_cpu_budget = borg.machine_to_normal(borg.unicore_cpu_budget(remaining))
            (feasible_b,) = numpy.digitize([normal_cpu_budget], self._budgets)

            if feasible_b == 0:
                break

            raw_plan = \
                plan_knapsack_multiverse(
                    tclass_weights_L,
                    augmented_tclass_rates_LAB[..., :feasible_b],
                    )

            # interpret the plan's first action
            (a, c) = raw_plan[0]
            planned_cpu_cost = self._budgets[c]

            if a >= S:
                solver = paused.pop(a - S)
                s = solver.s
                name = self._solver_names[s]
            else:
                s = a
                name = self._solver_names[s]
                solver = self._solvers[name](cnf_path, queue, uuid.uuid4())
                solver.s = s
                solver.cpu_cost = 0.0

            # don't waste our final seconds before the buzzer
            if normal_cpu_budget - planned_cpu_cost < self._budgets[0]:
                planned_cpu_cost = normal_cpu_budget
            else:
                planned_cpu_cost = planned_cpu_cost

            # be informative
            augmented_tclass_cmf_LAB = numpy.cumsum(augmented_tclass_rates_LAB, axis = -1)
            augmented_mean_cmf_AB = numpy.sum(tclass_weights_L[:, None, None] * augmented_tclass_cmf_LAB, axis = 0)
            subjective_rate = augmented_mean_cmf_AB[a, c]

            logger.info(
                "running %s@%i for %i with %i remaining (b = %.2f)",
                name,
                borg.normal_to_machine(solver.cpu_cost),
                borg.normal_to_machine(planned_cpu_cost),
                remaining.cpu_seconds,
                subjective_rate,
                )

            # ... and follow through
            solver.unpause_for(borg.normal_to_machine(planned_cpu_cost))

            running[solver._solver_id] = solver

            solver.cpu_budgeted = solver.cpu_cost + planned_cpu_cost

            if len(running) == cores:
                (solver_id, run_cpu_seconds, answer, terminated) = queue.get()

                borg.get_accountant().charge_cpu(run_cpu_seconds)

                solver = running.pop(solver_id)

                solver.cpu_cost += borg.machine_to_normal(run_cpu_seconds)

                if answer is not None:
                    break
                elif terminated:
                    failed.append(solver)
                else:
                    paused.append(solver)

        for process in paused + running.values():
            process.stop()

        return answer

borg.portfolios.named["borg-mix+class"] = BilevelPortfolio

