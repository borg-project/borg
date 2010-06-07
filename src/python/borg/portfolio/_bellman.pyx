"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import  numpy
cimport numpy

from cargo.log import  get_logger
from _models   cimport Predictor

log = get_logger(__name__)

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t

    void* malloc(size_t size)
    void  free  (void *ptr)

cdef struct BellmanCache:
    double        discount
    double*       costs
    double*       utilities
    size_t*       sizes
    size_t        M
    size_t        D
    unsigned int* history

cdef struct BellmanStackFrame:
    double* predictions

def compute_bellman_utility(
    model,
    unsigned int horizon,
    double budget,
    double discount,
    history,
    ):
    """
    Compute the expected utility of a state.
    """

    # cache action costs and outcome utilities
    cdef numpy.ndarray[double, ndim = 1, mode = "c"] costs       = numpy.empty(len(model.actions))
    cdef numpy.ndarray[double, ndim = 2, mode = "c"] utilities   = numpy.empty(history.shape)
    cdef numpy.ndarray[size_t, ndim = 1, mode = "c"] sizes = numpy.empty(len(model.actions), numpy.uint)

    for (i, action) in enumerate(model.actions):
        costs[i] = action.cost
        sizes[i] = len(action.outcomes)

        for (j, outcome) in enumerate(action.outcomes):
            utilities[i, j] = outcome.utility

    # set up the fixed-size stack
    cdef BellmanStackFrame* stack = <BellmanStackFrame*>malloc(horizon * sizeof(BellmanStackFrame))

    for i in xrange(horizon):
        stack[i].predictions = <double*>malloc(history.size * sizeof(double))

    # set up common data
    cdef numpy.ndarray[unsigned int, ndim = 2, mode = "c"] fancy_history = history
    cdef BellmanCache cache

#     cache.predictor = model.predictor
    cache.discount  = discount
    cache.costs     = <double*>costs.data
    cache.utilities = <double*>utilities.data
    cache.sizes     = <size_t*>sizes.data
    cache.M         = history.shape[0]
    cache.D         = history.shape[1]
    cache.history   = <unsigned int*>fancy_history.data

    # invoke the inner loop
    (utility, plan) = compute_bellman_utility_inner(model.predictor, &cache, stack, horizon, budget)

    # return the simple plan
    return (utility, [model.actions[i] for i in plan])

cdef compute_bellman_utility_inner(
    Predictor          predictor,
    BellmanCache*      cache,
    BellmanStackFrame* frame,
    unsigned int       horizon,
    double             budget,
    ):
    """
    Compute the expected utility of a state.
    """

    # base case
    if horizon == 0:
        return (0.0, [])

    # mise en place
    cdef size_t M = cache.M
    cdef size_t D = cache.D

    # predict outcomes at this state
    predictor.predict_raw(cache.history, frame.predictions)

    # used by the loop
    cdef double expected
    cdef double best_expected = -1.0
    cdef size_t best_action   = 0
    cdef double cost
    cdef double utility
    cdef size_t i
    cdef size_t j

    best_plan = None

    # for every action
    for i in xrange(cache.M):
        cost = cache.costs[i]

        if cost <= budget:
            expected = 0.0

            # for every outcome
            for j in xrange(cache.sizes[i]):
                utility = cache.utilities[i * D + j]

                if utility > 0.0:
                    expected += frame.predictions[i * D + j] * cache.discount**cost * utility
                else:
                    cache.history[i * D + j] += 1

                    (t_e, t_plan) = \
                        compute_bellman_utility_inner(
                            predictor,
                            cache,
                            frame + 1,
                            horizon - 1,
                            budget - cost,
                            )

                    cache.history[i * D + j] -= 1

                    expected += frame.predictions[i * D + j] * cache.discount**cost * t_e

            if expected >= best_expected:
                best_expected = expected
                best_action   = i
                best_plan     = t_plan

    if horizon >= 2:
        log.info(
            "%i: %f %s",
            horizon,
            best_expected,
            None if best_expected < 0.0 else best_action,
            )

    if best_expected < 0.0:
        return (0.0, [])
    else:
        return (best_expected, [best_action] + best_plan)

