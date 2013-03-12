#cython: profile=False
"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import borg

cimport cython
cimport libc.math
cimport numpy
cimport borg.statistics

logger = borg.get_logger(__name__, default_level = "INFO")

cdef extern from "math.h":
    double INFINITY

class Planner(object):
    """Discretizing dynamic-programming planner."""

    def __init__(self, compute_plan):
        self._compute_plan = compute_plan

    def plan(self, log_survival, log_weights = None):
        """Compute a plan."""

        log_survival_WSB = log_survival

        (W, _, _) = log_survival_WSB.shape

        if log_weights is None:
            log_weights_W = -numpy.ones(W) * numpy.log(W)
        else:
            log_weights_W = log_weights

        return self._compute_plan(log_survival_WSB, log_weights_W)

@cython.infer_types(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def knapsack_plan(log_survival, log_weights):
    """Compute a plan via dynamic programming."""

    # prepare
    cdef int W
    cdef int S
    cdef int B

    (W, S, B) = log_survival.shape

    # generate the value table and associated policy
    log_survival_swapped = numpy.asarray(log_survival.swapaxes(0, 1).swapaxes(1, 2), order = "C")

    cdef numpy.ndarray[double, ndim = 3] log_survival_SBW = log_survival_swapped
    cdef numpy.ndarray[double, ndim = 3] survival_SBW = numpy.exp(log_survival_swapped)
    cdef numpy.ndarray[double, ndim = 1] log_weights_W = numpy.asarray(log_weights, order = "C")
    cdef numpy.ndarray[double, ndim = 1] weights_W = numpy.exp(log_weights_W + libc.math.log(W))
    cdef numpy.ndarray[double, ndim = 2] values_B1W = numpy.ones((B + 1, W))
    cdef numpy.ndarray[int, ndim = 1] policy_s_B = numpy.empty(B, numpy.intc)
    cdef numpy.ndarray[int, ndim = 1] policy_c_B = numpy.empty(B, numpy.intc)

    cdef double post
    cdef double best_post
    cdef double v
    cdef int best_s
    cdef int best_c
    cdef int b
    cdef int s
    cdef int c
    cdef int w

    for b in xrange(1, B + 1):
        best_s = 0
        best_c = 0
        best_post = INFINITY

        for s in xrange(S):
            for c in xrange(b):
                post = 0.0

                for w in xrange(W):
                    v = weights_W[w] * survival_SBW[s, c, w] * values_B1W[b - c - 1, w]

                    post += v

                if post < best_post:
                    best_s = s
                    best_c = c
                    best_post = post

        for w in xrange(W):
            values_B1W[b, w] = survival_SBW[best_s, best_c, w] * values_B1W[b - best_c - 1, w]

        policy_s_B[b - 1] = best_s
        policy_c_B[b - 1] = best_c

    # build a plan from the policy
    plan = []
    b = B

    while b > 0:
        s = policy_s_B[b - 1]
        c = policy_c_B[b - 1]
        b -= c + 1

        plan.append((s, c))

    return plan

class KnapsackPlanner(Planner):
    """Discretizing dynamic-programming planner."""

    def __init__(self):
        Planner.__init__(self, knapsack_plan)

def streeter_plan(log_survival_WSB, log_weights_W):
    """Compute plan using Streeter's algorithm."""

    # prepare
    (W, S, B) = log_survival_WSB.shape

    # plan
    R = B
    plan = []
    log_plan_survival_W = numpy.zeros(W)

    while R > 0:
        log_post_survival_WSR = log_survival_WSB[..., :R] + log_plan_survival_W[:, None, None]

        f_plan = numpy.sum(1.0 - numpy.exp(log_plan_survival_W))
        f_post = numpy.sum(1.0 - numpy.exp(log_post_survival_WSR), axis = 0)
        flat_sb = numpy.argmax((f_post - f_plan) / numpy.arange(1, R + 1))
        (min_s, min_b) = action = numpy.unravel_index(flat_sb, f_post.shape)

        plan.append(action)

        log_plan_survival_W += log_post_survival_WSR[:, min_s, min_b]

        R -= min_b + 1

    # ...
    return plan

class StreeterPlanner(Planner):
    """
    Greedy approximate planner from Streeter et al.

    Extended to support instance weights.
    """

    def __init__(self):
        Planner.__init__(self, streeter_plan)

cdef struct PlannerState:
    int W
    int S
    int B
    double* belief_stack_BW
    double* log_survival_WSB

@cython.infer_types(True)
cdef object bellman_plan_full(PlannerState* this, int d):
    """Plan by solving the Bellman equation."""

    cdef double best_value = INFINITY
    cdef object best_plan = None
    cdef int best_s = -1
    cdef int best_b = -1

    for b in xrange(this.B - d):
        for s in xrange(this.S):
            next_d = d + b + 1
            survival = -INFINITY

            for w in xrange(this.W):
                failure = \
                    this.belief_stack_BW[d * this.W + w] \
                    + this.log_survival_WSB[w * this.S * this.B + s * this.B + b]

                if next_d < this.B:
                    this.belief_stack_BW[next_d * this.W + w] = failure

                survival = borg.statistics.log_plus(survival, failure)

            if next_d < this.B and survival > -INFINITY:
                for w in xrange(this.W):
                    this.belief_stack_BW[next_d * this.W + w] -= survival

                (value, plan) = bellman_plan_full(this, next_d)

                value += survival
            else:
                value = survival
                plan = []

            #print "d = {0}:".format(d), s, b, "[", ", ".join(["%.2f" % this.belief_stack_BW[next_d * this.W + w] for w in xrange(this.W)]), "]"

            if value <= best_value:
                best_value = value
                best_plan = plan
                best_s = s
                best_b = b

    #if d < 1:
        #print "{0}x{1} ({2:08.4f}) {3}={4}".format(best_s, best_b, best_value, "<" * d, d)

    return (best_value, [(best_s, best_b)] + best_plan)

class BellmanPlanner(object):
    """Discretizing optimal planner."""

    def plan(self, log_survival, log_weights = None):
        """Compute a plan."""

        # prepare
        log_survival = numpy.ascontiguousarray(log_survival, numpy.double)

        (W, S, B) = log_survival.shape

        if B == 0:
            return []

        if log_weights is None:
            log_weights_W = -numpy.ones(W) * numpy.log(W)
        else:
            log_weights_W = numpy.ascontiguousarray(log_weights, numpy.double)

        cdef numpy.ndarray log_survival_WSB = log_survival
        cdef numpy.ndarray belief_stack_BW = numpy.ones((B, W), numpy.double) * numpy.nan

        # compute the policy
        belief_stack_BW[0, :] = log_weights_W

        cdef PlannerState state

        state.W = W
        state.S = S
        state.B = B
        state.belief_stack_BW = <double*>belief_stack_BW.data
        state.log_survival_WSB = <double*>log_survival_WSB.data

        #print "starting to plan"
        #with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
            #print log_weights_W
        #print "..."

        (value, plan) = bellman_plan_full(&state, 0)

        ### heuristically reorder the plan
        #print "plan is)", plan

        #log_mean_fail_cmf_SB = numpy.logaddexp.reduce(log_survival_WSB + log_weights_W[:, None, None], axis = 0)

        #print "full survival function:"
        #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
            #print log_mean_fail_cmf_SB
        #print "marginal survival function:"
        #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
            #print log_mean_fail_cmf_SB

        #raise SystemExit()

        #def heuristic(pair):
            #(s, c) = pair

            #return log_mean_fail_cmf_SB[s, c] / (c + 1)

        #return sorted(plan, key = heuristic)

        return plan

class ReorderingPlanner(Planner):
    """Plan, then heuristically reorder."""

    def __init__(self, inner_planner):
        """Initialize."""

        def compute_plan(log_survival_WSB, log_weights_W):
            plan = inner_planner.plan(log_survival_WSB, log_weights_W)
            log_mean_fail_cmf_SB = numpy.logaddexp.reduce(log_survival_WSB + log_weights_W[:, None, None], axis = 0)

            def efficiency(pair):
                (s, c) = pair

                return log_mean_fail_cmf_SB[s, c] / (c + 1)

            return sorted(plan, key = efficiency)

        Planner.__init__(self, compute_plan)

class ReplanningPlanner(object):
    """Repeatedly replan."""

    def __init__(self, inner_planner):
        self._inner_planner = inner_planner

    def plan(self, log_survival, log_weights = None):
        # plan;
        # take first solver;
        # assume failure;
        # add to survival function;
        # repeat
        # XXX
        pass

class ResumptionPlanner(object):
    """Include solver resumption in planning."""

    def __init__(self, inner_planner):
        self._inner_planner = inner_planner

    def plan(self, log_survival, log_weights = None):
        # plan;
        # take first solver;
        # assume failure;
        # add to survival function;
        # repeat
        # XXX
        pass

default = ReorderingPlanner(KnapsackPlanner())

