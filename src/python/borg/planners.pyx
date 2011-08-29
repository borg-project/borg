"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import cargo

cimport numpy
cimport borg.statistics

logger = cargo.get_logger(__name__, default_level = "INFO")

cdef extern from "math.h":
    double INFINITY

class KnapsackMultiversePlanner(object):
    """Discretizing dynamic-programming planner."""

    def plan(self, log_survival, log_weights = None):
        """Compute a plan."""

        # prepare
        (W, S, B) = log_survival.shape

        log_survival_WSB = log_survival

        if log_weights is None:
            log_weights_W = -numpy.ones(W) * numpy.log(W)
        else:
            log_weights_W = log_weights

        # generate the value table and associated policy
        values_WB1 = numpy.zeros((W, B + 1))
        policy = {}

        for b in xrange(1, B + 1):
            region_WSb = log_survival_WSB[..., :b] + values_WB1[:, None, b - 1::-1]
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

            plan.append((s, c))

        # heuristically reorder the plan
        log_mean_fail_cmf_SB = numpy.logaddexp.reduce(log_survival_WSB + log_weights_W[:, None, None], axis = 0)

        def heuristic(pair):
            (s, c) = pair

            return log_mean_fail_cmf_SB[s, c] / (c + 1)

        plan = sorted(plan, key = heuristic)

        # ...
        return plan

cdef struct BellmanPlannerState:
    int W
    int S
    int B
    int* best_s_B
    int* best_b_B
    double* belief_stack_BW
    double* log_survival_WSB

cdef double bellman_planner_recurse(BellmanPlannerState state, int d):
    """Recurse, solving the Bellman equation."""

    cdef int W = state.W
    cdef int S = state.S
    cdef int B = state.B

    cdef int w
    cdef int s
    cdef int b

    cdef double best_value = INFINITY
    cdef int best_s = -1
    cdef int best_b = -1

    for b in xrange(state.B - d):
        for s in xrange(S):
            next_d = d + b + 1
            survival = -INFINITY

            for w in xrange(W):
                failure = state.belief_stack_BW[d * B + w] + state.log_survival_WSB[w * W * S + s * S + b]

                if next_d < B:
                    state.belief_stack_BW[next_d * B + w] = failure

                survival = borg.statistics.log_plus(survival, failure)

            if next_d < B:
                for w in xrange(W):
                    state.belief_stack_BW[next_d * B + w] -= survival

                value = survival + bellman_planner_recurse(state, next_d)
            else:
                value = survival

            if d == 0:
                print "* {0}@{1}: {2}".format(s, b, value)

            if value < best_value:
                best_value = value
                best_s = s
                best_b = b

    if d < 6:
        print "{0}{1} {2}@{3} ({4})".format(d, ">" * d, best_s, best_b, best_value)

    state.best_s_B[d] = best_s
    state.best_b_B[d] = best_b

    return best_value

class BellmanPlanner(object):
    """Discretizing optimal planner."""

    def plan(self, log_survival, log_weights = None):
        """Compute a plan."""

        # prepare
        (W, S, B) = log_survival.shape

        if B == 0:
            return []

        cdef numpy.ndarray log_survival_WSB = log_survival

        if log_weights is None:
            log_weights_W = -numpy.ones(W) * numpy.log(W)
        else:
            log_weights_W = log_weights

        cdef numpy.ndarray best_s_B = numpy.ones(B, numpy.intc) * -1
        cdef numpy.ndarray best_b_B = numpy.ones(B, numpy.intc) * -1
        cdef numpy.ndarray belief_stack_BW = numpy.ones((B, W), numpy.double) * numpy.nan

        # compute the policy
        logger.info("computing an optimal plan")

        belief_stack_BW[0, :] = log_weights_W

        cdef BellmanPlannerState state

        state.W = W
        state.S = S
        state.B = B
        state.best_s_B = <int*>best_s_B.data
        state.best_b_B = <int*>best_b_B.data
        state.belief_stack_BW = <double*>belief_stack_BW.data
        state.log_survival_WSB = <double*>log_survival_WSB.data

        bellman_planner_recurse(state, 0)

        # build a plan from the policy
        plan = []
        b = 0

        while b < B:
            s = self._best_s_B[b]
            c = self._best_b_B[b]
            b += c + 1

            plan.append((s, c))

        # ...
        return plan

    def _recurse(self, d):
        """Recurse, solving the Bellman equation."""

        (W, S, B) = self._log_survival_WSB.shape

        best_value = numpy.inf
        best_s = None
        best_b = None

        for b in xrange(B - d):
            for s in xrange(S):
                next_d = d + b + 1
                survival = -numpy.inf

                for w in xrange(W):
                    failure = self._belief_stack_BW[d, w] + self._log_survival_WSB[w, s, b]

                    if next_d < B:
                        self._belief_stack_BW[next_d, w] = failure

                    survival = borg.statistics.log_plus(survival, failure)

                if next_d < B:
                    self._belief_stack_BW[next_d, :] -= survival

                    value = survival + self._recurse(next_d)
                else:
                    value = survival

                if d == 0:
                    print "* {0}@{1}: {2}".format(s, b, value)

                if value < best_value:
                    best_value = value
                    best_s = s
                    best_b = b

        if d < 6:
            print "{0}{1} {2}@{3} ({4})".format(d, ">" * d, best_s, best_b, best_value)

        self._best_s_B[d] = best_s
        self._best_b_B[d] = best_b

        return best_value

