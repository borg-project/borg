#cython: profile=False
"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import borg

cimport cython
cimport numpy
cimport borg.statistics

logger = borg.get_logger(__name__, default_level = "INFO")

cdef extern from "math.h":
    double INFINITY

class KnapsackMultiversePlanner(object):
    """Discretizing dynamic-programming planner."""

    def plan(self, log_survival, log_weights = None):
        """Compute a plan."""

        log_survival_WSB = log_survival

        (W, _, _) = log_survival_WSB.shape

        if log_weights is None:
            log_weights_W = -numpy.ones(W) * numpy.log(W)
        else:
            log_weights_W = log_weights

        return knapsack_plan(log_survival_WSB, log_weights_W)

def knapsack_plan(log_survival_WSB, log_weights_W):
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

    #print "plan is:", plan
    #print "full survival function:"
    #with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
        #print numpy.exp(log_survival_WSB)
    #print "marginal survival function:"
    #with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
        #print numpy.exp(log_mean_fail_cmf_SB)

    #raise SystemExit()

    # ...
    return plan

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

        ## heuristically reorder the plan
        print "plan is:", plan

        log_mean_fail_cmf_SB = numpy.logaddexp.reduce(log_survival_WSB + log_weights_W[:, None, None], axis = 0)

        print "full survival function:"
        with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
            print log_mean_fail_cmf_SB
        print "marginal survival function:"
        with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
            print log_mean_fail_cmf_SB

        raise SystemExit()

        #def heuristic(pair):
            #(s, c) = pair

            #return log_mean_fail_cmf_SB[s, c] / (c + 1)

        #return sorted(plan, key = heuristic)

        return plan

