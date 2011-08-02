"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import math
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
        log_weights_W = posterior.get_weights()
        log_fail_cmf_WSB = self.get_fail_cmf_matrix(posterior, remaining)

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

        ## heuristically reorder the plan
        #mean_rates_SB = numpy.sum(tclass_weights_W[:, None, None] * tclass_rates_WSB, axis = 0)
        #mean_cmf_SB = numpy.cumsum(mean_rates_SB, axis = -1)

        #def heuristic((s, c)):
            #return mean_cmf_SB[s, c] / (c + 1)

        #plan = sorted(plan, key = heuristic, reverse = True)

        # ...
        return plan

    def get_fail_cmf_matrix(self, posterior, remaining):
        """Populate a failure-CMF matrix via discretization."""

        W = posterior.components
        S = len(self._solver_index)
        B = int(math.floor(remaining / self._budget_interval))

        log_fail_cmf_WSB = numpy.empty((W, S, B))

        for b in xrange(B):
            budget = self._budget_interval * (b + 1)

            for s in xrange(S):
                for w in xrange(W):
                    # XXX avoid the exponentiation
                    p = numpy.exp(posterior.get_log_cdf(w, s, budget))

                    if p == 1.0:
                        log_fail_cmf_WSB[w, s, b] = -numpy.inf
                    else:
                        log_fail_cmf_WSB[w, s, b] = numpy.log(1.0 - p)

        return log_fail_cmf_WSB

