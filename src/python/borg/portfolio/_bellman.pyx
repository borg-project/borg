"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import  numpy
cimport numpy

from cargo.log import  get_logger
from _models   cimport Predictor, SlowPredictor

log = get_logger(__name__)

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t

    void* malloc(size_t size)
    void  free  (void *ptr)

cdef struct BellmanCache:
    double        discount
    double*       costs
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
    cdef numpy.ndarray[double, ndim = 1, mode = "c"] costs = numpy.empty(len(model.actions))

    for (i, action) in enumerate(model.actions):
        costs[i] = action.cost

        if len(action.outcomes) != 2:
            raise RuntimeError("can only generate a sequence plan for two-outcome worlds")
        if action.outcomes[0].utility != 1.0:
            raise RuntimeError("first outcome assumed to have utility = 1.0")
        if action.outcomes[1].utility > 0.0:
            raise RuntimeError("second outcome assumed to have zero utility")

    # set up the fixed-size stack
    cdef BellmanStackFrame* stack = <BellmanStackFrame*>malloc(horizon * sizeof(BellmanStackFrame))

    for i in xrange(horizon):
        stack[i].predictions = <double*>malloc(history.size * sizeof(double))

    # set up common data
    cdef numpy.ndarray[unsigned int, ndim = 2, mode = "c"] fancy_history = history
    cdef BellmanCache cache

    cache.discount  = discount
    cache.costs     = <double*>costs.data
    cache.M         = M = history.shape[0]
    cache.D         = D = history.shape[1]
    cache.history   = <unsigned int*>fancy_history.data

    # grab the predictor
    if model.predictor is None:
        predictor = SlowPredictor(model.predict, M, D)
    else:
        predictor = model.predictor

    # invoke the inner loop
    (expected, plan) = compute_bellman_expected_inner(predictor, &cache, stack, horizon, budget)

    # return the simple plan
    return (expected, [model.actions[i] for i in plan])

cdef compute_bellman_expected_inner(
    Predictor          predictor,
    BellmanCache*      cache,
    BellmanStackFrame* frame,
    unsigned int       horizon,
    double             budget,
    ):
    """
    Compute the expected utility of a state.
    """

    # generate predictions for this state
    predictor.predict_raw(cache.history, frame.predictions)

    # used by the loop
    cdef double expected
    cdef double best_expected = -1.0
    cdef size_t best_action   = 0
    cdef double cost
    cdef size_t i
    cdef double t_e

    best_plan = []

    # for every action
    for i in xrange(cache.M):
        cost = cache.costs[i]

        if cost <= budget:
            expected = frame.predictions[i * 2] * cache.discount**cost

            if horizon > 1:
                cache.history[i * 2 + 1] += 1

                (t_e, t_plan) = \
                    compute_bellman_expected_inner(
                        predictor,
                        cache,
                        frame + 1,
                        horizon - 1,
                        budget - cost,
                        )

                cache.history[i * 2 + 1] -= 1

                expected += frame.predictions[i * 2 + 1] * cache.discount**cost * t_e
            else:
                t_plan = []

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

