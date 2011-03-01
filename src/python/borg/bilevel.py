"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import resource
import itertools
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def plan_knapsack_multiverse(model, failures, features, budgets, remaining):
    """
    Generate an optimal plan for a static set of possible probability tables.

    Budgets *must* be in ascending order, spaced by a fixed interval from zero.
    """

    # examine feasible budgets only
    feasible = budgets

    for (b, budget) in enumerate(budgets):
        if budget > remaining:
            feasible = budgets[:b]

            break

    if not feasible:
        return ([], 1.0)

    # generate model predictions
    (mean_rates_SB, tclass_weights_W, tclass_rates_WSB) = model.predict(failures, features)
    mean_cmf_SB = numpy.cumsum(mean_rates_SB, axis = -1)

    # convert them to log space
    log_weights_W = numpy.log(tclass_weights_W)

    #with borg.portfolios.fpe_handling(divide = "ignore"):
    log_fail_cmf_WSB = numpy.log(1.0 - numpy.cumsum(tclass_rates_WSB, axis = -1))

    # generate the value table and associated policy
    (W, S, B) = log_fail_cmf_WSB.shape
    values_WB = numpy.zeros((W, B + 1))
    policy = {}

    for b in xrange(1, len(feasible) + 1):
        region_WSb = log_fail_cmf_WSB[..., :b] + values_WB[:, None, b - 1::-1]
        weighted_Sb = numpy.logaddexp.reduce(region_WSb + log_weights_W[:, None, None], axis = 0)
        i = numpy.argmin(weighted_Sb)
        (s, c) = numpy.unravel_index(i, weighted_Sb.shape)

        values_WB[:, b] = region_WSb[:, s, c]
        policy[b] = (s, c)

    # compute our expectation of plan failure
    weighted_values_W = log_weights_W + values_WB[:, len(feasible)]
    expectation = numpy.sum(numpy.exp(weighted_values_W))

    # build a plan from the policy
    plan = []
    b = len(feasible)

    while b > 0:
        (_, c) = action = policy[b]
        b -= c + 1

        plan.append(action)

    # heuristically reorder the plan
    plan = sorted(plan, key = lambda (s, c): -mean_cmf_SB[s, c] / budgets[c])

    # ...
    return (plan, expectation)

def plan_knapsack_multiverse_speculative(model, failures, budgets, remaining):
    """Multiverse knapsack planner with speculative reordering."""

    # evaluate actions via speculative replanning
    (mean_rates_SB, _, _) = model.predict(failures)
    mean_fail_cmf_SB = numpy.cumprod(1.0 - mean_rates_SB, axis = -1)
    (S, B) = mean_rates_SB.shape
    speculation = []

    for (s, b) in itertools.product(xrange(S), xrange(B)):
        if budgets[b] <= remaining:
            p_f = mean_fail_cmf_SB[s, b]
            (post_plan, post_p_f) = \
                plan_knapsack_multiverse(
                    model,
                    failures + [(s, b)],
                    budgets,
                    remaining - budgets[b],
                    )
            expectation = p_f * post_p_f

            logger.info(
                "action %s p_f = %.2f; post-p_f = %.2f; expectation = %.4f",
                (s, b),
                p_f,
                post_p_f,
                expectation,
                )

            speculation.append(([(s, b)] + post_plan, expectation))

    # move the most useful action first
    if len(speculation) > 0:
        (plan, expectation) = min(speculation, key = lambda (_, e): e)

        return (plan, expectation)
    else:
        return ([], 1.0)

class BilevelPortfolio(object):
    """Bilevel mixture-model portfolio."""

    def __init__(self, solvers, train_paths):
        # build action set
        self._solvers = solvers
        self._budgets = budgets = [(b + 1) * 200.0 for b in xrange(25)]

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

        # gather oracle knowledge
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
        total_run_cost = features_cost
        failures = []
        answer = None

        while True:
            ## XXX for informational output only
            (predicted_SB, tclass_weights_L, tclass_rates_LSB) = self._model.predict(failures, features)
            predicted_cmf_SB = numpy.cumsum(predicted_SB, axis = -1)
            tclass_cmf_LSB = numpy.cumsum(tclass_rates_LSB, axis = -1)

            #with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 240):
                #for l in xrange(tclass_weights_L.size):
                    #print "conditional CMF for tclass {0} (weight {1:.2f}):".format(l, tclass_weights_L[l])
                    #print cargo.pretty_probability_matrix(tclass_cmf_LSB[l])

            # generate a plan
            total_cost = total_run_cost
            (raw_plan, _) = \
                plan_knapsack_multiverse(
                    self._model,
                    failures,
                    features,
                    self._budgets,
                    budget - total_cost,
                    )
            plan = [(self._solver_names[s], self._budgets[c]) for (s, c) in raw_plan]

            logger.detail("plan: %s", " -> ".join(map("{0[0]}@{0[1]:.0f}".format, plan)))

            if not plan:
                break

            # don't waste the final seconds before the buzzer
            (name, planned_cost) = plan[0]

            if budget - total_cost - planned_cost < self._budgets[0]:
                max_cost = budget - total_cost
            else:
                max_cost = planned_cost

            logger.info(
                "taking %s@%i with %i remaining (b = %.2f; p = %.2f)",
                name,
                max_cost,
                budget - total_cost,
                predicted_cmf_SB[self._solver_name_index[name], self._budget_index[planned_cost]],
                true_cmf[self._solver_name_index[name], self._budget_index[planned_cost]],
                )

            # run the first step in the plan
            solver = self._solvers[name]
            (run_cost, answer) = solver(cnf_path, max_cost)
            total_run_cost += run_cost

            if answer is not None:
                break
            else:
                failures.append((self._solver_name_index[name], self._budget_index[planned_cost]))

        logger.info("answer %s with total cost %.2f", answer, total_cost)

        #if answer is None:
            #raise SystemExit()

        return (total_cost, answer)

borg.portfolios.named["bilevel"] = BilevelPortfolio

