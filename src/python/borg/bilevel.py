"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import resource
import itertools
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
    threshold = 0.10

    if numpy.all(mean_cmf_SB <= threshold):
        logger.warning("not filtering tclasses; no entries in mean exceed %.2f", threshold)
    else:
        tclass_rates_WSB[:, mean_cmf_SB[:, :1] <= threshold] = 1e-8
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
        self._budgets = budgets = [(b + 1) * 100.0 for b in xrange(50)]

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

    def __call__(self, cnf_path, budget):
        logger.info("solving %s", os.path.basename(cnf_path))

        # track CPU time expenditure
        previous_utime = resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # gather oracle knowledge, if any
        (true_successes, true_attempts) = \
            borg.models.outcome_matrices_from_paths(
                self._solver_name_index,
                self._budgets,
                [cnf_path],
                )
        true_rates = true_successes[0] / true_attempts[0, ..., None]
        true_cmf = 1.0 - numpy.cumprod(1.0 - true_rates, axis = -1)

        logger.info("true CMF:\n%s", cargo.pretty_probability_matrix(true_cmf))

        # obtain features
        csv_path = cnf_path + ".features.csv"

        if os.path.exists(csv_path):
            features = numpy.recfromcsv(csv_path).tolist()
        else:
            (_, features) = borg.features.get_features_for(cnf_path)

        features_cost = features[0]

        # select a solver
        total_cost = features_cost
        failed = []
        paused = []
        states = []
        answer = None

        while True:
            # obtain model predictions
            (tclass_weights_L, tclass_rates_LSB) = self._model.predict(failed + paused, features)
            (L, S, B) = tclass_rates_LSB.shape

            # prepare augmented PMF matrix
            augmented_tclass_arrays = [tclass_rates_LSB]

            for (s, c) in paused:
                paused_tclass_LB = numpy.zeros((L, B))

                paused_tclass_LB[:, :B - c - 1] = tclass_rates_LSB[:, s, c + 1:]
                paused_tclass_LB /= 1.0 - numpy.sum(tclass_rates_LSB[:, s, :c + 1], axis = -1)[..., None]

                augmented_tclass_arrays.append(paused_tclass_LB[:, None, :])

            augmented_tclass_rates_LAB = numpy.hstack(augmented_tclass_arrays)

            # update cost tracking
            utime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
            total_cost += utime - previous_utime
            previous_utime = utime

            # make a plan...
            (feasible_b,) = numpy.digitize([budget - total_cost], self._budgets)

            if feasible_b == 0:
                break

            raw_plan = \
                plan_knapsack_multiverse(
                    tclass_weights_L,
                    augmented_tclass_rates_LAB[..., :feasible_b],
                    )

            # interpret the plan's first action
            (a, c) = raw_plan[0]
            planned_cost = self._budgets[c]

            if a >= S:
                (solver, solver_total) = states.pop(a - S)
                (s, _) = paused.pop(a - S)
                name = self._solver_names[s]
            else:
                s = a
                name = self._solver_names[s]
                solver = self._solvers[name]
                solver_total = 0.0

            if budget - total_cost - planned_cost < self._budgets[0]:
                max_cost = budget - total_cost
            else:
                max_cost = planned_cost

            # be informative
            augmented_tclass_cmf_LAB = numpy.cumsum(augmented_tclass_rates_LAB, axis = -1)
            augmented_mean_cmf_AB = numpy.sum(tclass_weights_L[:, None, None] * augmented_tclass_cmf_LAB, axis = 0)
            subjective_rate = augmented_mean_cmf_AB[a, c]

            #for l in xrange(L):
                #print "augmented conditional CMF for tclass {0} (weight {1:.2f}):".format(l, tclass_weights_L[l])
                #print cargo.pretty_probability_matrix(augmented_tclass_cmf_LAB[l])

            logger.info(
                "running %s@%i for %i with %i remaining (b = %.2f)",
                name,
                solver_total,
                max_cost,
                budget - total_cost,
                subjective_rate,
                )

            # ... and follow through
            (run_cost, answer, resume) = solver(cnf_path, max_cost)
            total_cost += run_cost

            if answer is not None:
                break
            elif resume is None:
                failed.append((self._solver_name_index[name], self._budget_index[planned_cost]))
            else:
                new_solver_total = solver_total + run_cost
                (total_b,) = numpy.digitize([new_solver_total], self._budgets)

                paused.append((self._solver_name_index[name], total_b - 1))
                states.append((resume, new_solver_total))

        logger.info("answer %s with total cost %.2f", answer, total_cost)

        return (total_cost, answer, None)

borg.portfolios.named["borg-mix+class>>0.1"] = BilevelPortfolio
#borg.portfolios.named["borg-mix+class"] = BilevelPortfolio

