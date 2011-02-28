"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import resource
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def plan_knapsack_multiverse(model, failures, budgets, remaining):
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
    (_, tclass_weights_L, tclass_rates_LSB) = model.predict(failures)

    # convert them to log space
    log_weights_W = numpy.log(tclass_weights_L)

    with borg.portfolios.fpe_handling(divide = "ignore"):
        log_rates_WSB = numpy.log(1.0 - tclass_rates_LSB)

    # generate the value table and associated policy
    (W, S, B) = log_rates_WSB.shape
    values_WB = numpy.zeros((W, B + 1))
    policy = {}

    for b in xrange(1, len(feasible) + 1):
        region_WSb = log_rates_WSB[..., :b] + values_WB[:, None, b - 1::-1]
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

    # ...
    return (plan, expectation)

def plan_knapsack_multiverse_speculative(model, failures, budgets, remaining):
    """Multiverse knapsack planner with speculative reordering."""

    # compute an initial plan
    (base_plan, base_expectation) = plan_knapsack_multiverse(model, failures, budgets, remaining)

    if len(base_plan) <= 1:
        return (base_plan, base_expectation)

    # reorder via speculative replanning
    (mean_rates_SB, _, _) = model.predict(failures) # XXX already called in planning above
    speculation = []

    for (i, (s, c)) in enumerate(base_plan):
        p_f = 1.0 - mean_rates_SB[s, c]
        (post_plan, post_p_f) = \
            plan_knapsack_multiverse(
                model,
                failures + [(s, c)],
                budgets,
                remaining - budgets[c],
                )
        expectation = p_f * post_p_f

        logger.debug(
            "action %s p_f = %.2f, post-p_f = %.2f, expectation = %.4f",
            (s, c),
            p_f,
            post_p_f,
            expectation,
            )

        speculation.append(([base_plan[i]] + post_plan, expectation))

    # move the most useful action first
    (plan, _) = max(speculation, key = lambda (_, e): e)

    return (plan, base_expectation)

class BilevelPortfolio(object):
    """Bilevel mixture-model portfolio."""

    def __init__(self, solvers, train_paths):
        # build action set
        self._solvers = solvers
        self._budgets = budgets = [(b + 1) * 200.0 for b in xrange(25)]

        # acquire running time data
        self._solver_rindex = dict(enumerate(solvers))
        self._solver_index = dict(map(reversed, self._solver_rindex.items()))
        self._budget_rindex = dict(enumerate(budgets))
        self._budget_index = dict(map(reversed, self._budget_rindex.items()))

        (successes, attempts) = borg.models.counts_from_paths(self._solver_index, self._budget_index, train_paths)

        logger.info("solvers: %s", self._solver_rindex)

        # fit our model
        self._model = borg.models.BilevelModel(successes, attempts)

    def __call__(self, cnf_path, budget):
        logger.info("solving %s", os.path.basename(cnf_path))

        # gather oracle knowledge
        (runs,) = borg.portfolios.get_task_run_data([cnf_path]).values()
        (oracle_history, oracle_counts, _) = \
            borg.portfolios.action_rates_from_runs(
                self._solver_index,
                self._budget_index,
                runs.tolist(),
                )
        true_rates = oracle_history / oracle_counts

        logger.info("true probabilities:\n%s", cargo.pretty_probability_matrix(true_rates))

        # select a solver
        total_run_cost = 0.0
        failures = []
        answer = None

        while True:
            # XXX for informational output only
            (predicted, tclass_weights_L, tclass_rates_LSB) = self._model.predict(failures)

            with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 240):
                for l in xrange(tclass_weights_L.size):
                    print "probabilities for tclass {0} (weight {1:.2f}):".format(l, tclass_weights_L[l])
                    print cargo.pretty_probability_matrix(tclass_rates_LSB[l])

            # generate a plan
            total_cost = total_run_cost
            (raw_plan, _) = \
                plan_knapsack_multiverse_speculative(
                    self._model,
                    failures,
                    self._budgets,
                    budget - total_cost,
                    )
            plan = [(self._solver_rindex[s], self._budgets[c]) for (s, c) in raw_plan]

            logger.info("plan: %s", " -> ".join(map("{0[0]}@{0[1]:.0f}".format, plan)))

            if not plan:
                break

            # don't waste our closing seconds
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
                predicted[self._solver_index[name], self._budget_index[planned_cost]],
                true_rates[self._solver_index[name], self._budget_index[planned_cost]],
                )

            # run the first step in the plan
            solver = self._solvers[name]
            (run_cost, answer) = solver(cnf_path, max_cost)
            total_run_cost += run_cost

            if answer is not None:
                break
            else:
                failures.append((self._solver_index[name], self._budget_index[planned_cost]))

        logger.info("answer %s with total cost %.2f", answer, total_cost)

        if answer is None:
            raise SystemExit()

        return (total_cost, answer)

borg.portfolios.named["bilevel"] = BilevelPortfolio

