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

def knapsack_plan(log_survival_WSB, log_weights_W, give_log_fail = False):
    """Compute a plan."""

    # prepare
    (W, S, B) = log_survival_WSB.shape

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
    if give_log_fail:
        return (plan, numpy.logaddexp.reduce(log_weights_W + values_WB1[:, B]))
    else:
        return plan

@cython.profile(False)
cdef double log_plus(double x, double y):
    """
    Return log(x + y) given log(x) and log(y); see [1].

    [1] Digital Filtering Using Logarithmic Arithmetic. Kingsbury and Rayner, 1970.
    """

    if x == -INFINITY and y == -INFINITY:
        return -INFINITY
    elif x >= y:
        return x + libc.math.log(1.0 + libc.math.exp(y - x))
    else:
        return y + libc.math.log(1.0 + libc.math.exp(x - y))

@cython.infer_types(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def knapsack_plan_fast(log_survival, log_weights):
    """Compute a plan."""

    # prepare
    cdef int W
    cdef int S
    cdef int B

    (W, S, B) = log_survival.shape

    # generate the value table and associated policy
    log_survival_swapped = numpy.asarray(log_survival.swapaxes(0, 1).swapaxes(1, 2), order = "C")

    cdef numpy.ndarray[double, ndim = 3] log_survival_SBW = log_survival_swapped
    cdef numpy.ndarray[double, ndim = 1] log_weights_W = numpy.asarray(log_weights, order = "C")
    cdef numpy.ndarray[double, ndim = 2] values_B1W = numpy.zeros((B + 1, W))
    cdef numpy.ndarray[int, ndim = 1] policy_s_B = numpy.empty(B, numpy.intc)
    cdef numpy.ndarray[int, ndim = 1] policy_c_B = numpy.empty(B, numpy.intc)

    cdef double post_log
    cdef double best_post_log
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
        best_post_log = INFINITY

        for s in xrange(S):
            for c in xrange(b):
                post_log = -INFINITY

                for w in xrange(W):
                    v = log_weights_W[w] + log_survival_SBW[s, c, w] + values_B1W[b - c - 1, w]

                    post_log = log_plus(post_log, v)

                if post_log < best_post_log:
                    best_s = s
                    best_c = c
                    best_post_log = post_log

        for w in xrange(W):
            values_B1W[b, w] = log_survival_SBW[best_s, best_c, w] + values_B1W[b - best_c - 1, w]

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
        #Planner.__init__(self, knapsack_plan)
        Planner.__init__(self, knapsack_plan_fast)

def max_length_knapsack_plan(max_length, log_survival_WSB, log_weights_W):
    """Compute a plan subject to a maximum-length constraint."""

    # XXX
    log_survival_SB = numpy.logaddexp.reduce(log_survival_WSB + log_weights_W[:, None, None], axis = 0)

    with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 240, threshold = 1000000):
        print 1.0 - numpy.exp(log_survival_SB)

    # prepare
    (W, S, B) = log_survival_WSB.shape
    C = B + 1
    L = max_length

    # generate the value table and associated policy
    values_WCL = numpy.zeros((W, C, L))
    policy = {}

    for b in xrange(1, B + 1):
        for l in xrange(L):
            if l == 0:
                region_WSb = log_survival_WSB[..., :b]
            else:
                region_WSb = log_survival_WSB[..., :b] + values_WCL[:, None, b - 1::-1, l - 1]

            weighted_Sb = numpy.logaddexp.reduce(region_WSb + log_weights_W[:, None, None], axis = 0)
            i = numpy.argmin(weighted_Sb)
            (s, c_) = numpy.unravel_index(i, weighted_Sb.shape)

            values_WCL[:, b, l] = region_WSb[:, s, c_]
            policy[(b, l)] = (s, c_)

    # build a plan from the policy
    plan = []
    b = B
    l = L - 1

    while b > 0 and l >= 0:
        (s, c) = policy[(b, l)]
        b -= c + 1
        l -= 1

        plan.append((s, c))

    return plan

class MaxLengthKnapsackPlanner(Planner):
    """Discretizing dynamic-programming planner."""

    def __init__(self, max_length):
        def plan(log_survival, log_weights):
            return max_length_knapsack_plan(max_length, log_survival, log_weights)

        Planner.__init__(self, plan)

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
        #print
        #print "...", R
        #print 1.0 - numpy.exp(log_post_survival_WSR)
        #print f_plan
        #print f_post
        #print min_s, min_b

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

#@cython.infer_types(True)
#cdef object bellman_plan_hybrid(PlannerState* this, int d, int threshold):
    #"""Plan by solving the Bellman equation."""

    #cdef double best_value = INFINITY
    #cdef object best_plan = None
    #cdef int best_s = -1
    #cdef int best_b = -1

    #for b in xrange(this.B - d):
        #for s in xrange(this.S):
            #next_d = d + b + 1
            #survival = -INFINITY

            #for w in xrange(this.W):
                #failure = \
                    #this.belief_stack_BW[d * this.W + w] \
                    #+ this.log_survival_WSB[w * this.S * this.B + s * this.B + b]

                #if next_d < this.B:
                    #this.belief_stack_BW[next_d * this.W + w] = failure

                #survival = borg.statistics.log_plus(survival, failure)

            #if next_d < this.B and survival > -INFINITY:
                #for w in xrange(this.W):
                    #this.belief_stack_BW[next_d * this.W + w] -= survival

                #if threshold == 0:
                    #(value, plan) = bellman_plan_hybrid(this, next_d, threshold - 1)
                #else:
                    #(value, plan) = bellman_plan_hybrid(this, next_d, threshold - 1)

                #value += survival
            #else:
                #value = survival
                #plan = []

            #if value <= best_value:
                #best_value = value
                #best_plan = plan
                #best_s = s
                #best_b = b

    #if d < 5:
        #print "{0}x{1} ({2:08.4f}) {3}={4}".format(best_s, best_b, best_value, "<" * d, d)

    #return (best_value, [(best_s, best_b)] + best_plan)

def plan_log_survival(plan, log_survival_WSB, log_weights_W):
    """Compute the survival probability of a plan."""

    plan_log_survival_W = numpy.copy(log_weights_W)

    for (s, b) in plan:
        plan_log_survival_W += log_survival_WSB[:, s, b]

    return numpy.logaddexp.reduce(plan_log_survival_W)

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

