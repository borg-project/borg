"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import resource
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def knapsack_plan_multiverse(weights, rates, solver_rindex, budgets, remaining):
    """
    Generate an optimal plan for a static set of possible probability tables.

    Budgets *must* be in ascending order and *must* be spaced by a fixed
    interval from zero.
    """

    # examine feasible budgets only
    for (b, budget) in enumerate(budgets):
        if budget > remaining:
            budgets = budgets[:b]

            break

    if not budgets:
        return []

    # generate the value table and associated policy
    log_weights_W = numpy.log(weights)

    with borg.portfolios.fpe_handling(divide = "ignore"):
        log_rates_WSB = numpy.log(1.0 - rates)

    (W, S, B) = log_rates_WSB.shape
    values_WB = numpy.zeros((W, B + 1))
    policy = {}

    for b in xrange(1, len(budgets) + 1):
        region_WSb = log_rates_WSB[..., :b] + values_WB[:, None, b - 1::-1]
        weighted_Sb = numpy.logaddexp.reduce(region_WSb + log_weights_W[:, None, None], axis = 0)
        i = numpy.argmin(weighted_Sb)

        #with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 240):
            #print "values_WB:"
            #print values_WB
            #print "region_WSb:"
            #print region_WSb
            #print "weighted_Sb:"
            #print weighted_Sb

        (s, c) = numpy.unravel_index(i, weighted_Sb.shape)

        values_WB[:, b] = region_WSb[:, s, c]
        policy[b] = (s, c)

    # build a plan from the policy
    plan = []
    b = len(budgets)

    while b > 0:
        (_, c) = action = policy[b]
        b -= c + 1

        plan.append(action)

        #print "optimal action is", solver_rindex[action[0]], budgets[c]

    # heuristically reorder the plan
    #reordered = sorted(plan, key = lambda (s, c): -rates[s, c] / budgets[c])
    reordered = sorted(plan, key = lambda (s, c): c)

    # translate and return it
    return [(solver_rindex[s], budgets[c]) for (s, c) in reordered]

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

        # fit our model
        self._model = borg.models.BilevelModel(successes, attempts)

    def __call__(self, cnf_path, budget):
        # gather oracle knowledge
        (runs,) = borg.portfolios.get_task_run_data([cnf_path]).values()
        (oracle_history, oracle_counts, _) = \
            borg.portfolios.action_rates_from_runs(
                self._solver_index,
                self._budget_index,
                runs.tolist(),
                )
        true_rates = oracle_history / oracle_counts

        print "true probabilities:"
        print cargo.pretty_probability_matrix(true_rates)

        # select a solver
        total_run_cost = 0.0
        failures = []
        answer = None

        logger.info("solving %s", os.path.basename(cnf_path))

        while True:
            # compute marginal probabilities of success
            (predicted, tclass_weights_L, tclass_rates_LSB) = self._model.predict(failures)

            # generate a plan
            total_cost = total_run_cost

            # XXX
            plan = knapsack_plan_multiverse(tclass_weights_L, tclass_rates_LSB, self._solver_rindex, self._budgets, budget - total_cost)

            if not plan:
                break

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
                predicted[self._solver_index[name], self._budget_index[planned_cost]],
                true_rates[self._solver_index[name], self._budget_index[planned_cost]],
                )

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

