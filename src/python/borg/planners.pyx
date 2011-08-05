"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy

class KnapsackMultiversePlanner(object):
    """Discretizing dynamic-programming planner."""

    def __init__(self, solver_index, budget_interval):
        """Initialize."""

        self._solver_index = solver_index
        self._budget_interval = budget_interval

    def plan(self, posterior, remaining):
        """Compute a plan."""

        # convert model predictions to a log CMF
        budgets_B = numpy.r_[self._budget_interval:remaining + 1:self._budget_interval]

        log_weights_W = posterior.get_weights()
        log_cmf_WSB = posterior.get_log_cdf_array(budgets_B)
        log_fail_cmf_WSB = numpy.log(1.0 - numpy.exp(log_cmf_WSB)) # XXX avoid exponentiation

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
            (s, c) = policy[b]
            b -= c + 1

            plan.append((s, self._budget_interval * (c + 1)))

        # heuristically reorder the plan
        log_mean_fail_cmf_SB = numpy.logaddexp.reduce(log_fail_cmf_WSB + log_weights_W[:, None, None], axis = 0)

        def heuristic(pair):
            (s, budget) = pair
            c = int(budget / self._budget_interval) - 1

            assert c >= 0

            return log_mean_fail_cmf_SB[s, c] / (c + 1)

        plan = sorted(plan, key = heuristic)

        # ...
        return plan

