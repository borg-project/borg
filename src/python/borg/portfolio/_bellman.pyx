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
    void free(void *ptr)
    void *malloc(size_t size)

cdef struct BellmanStackEntry:
#     double e
#     double best_e
#     Py_ssize_t best_action
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
    cdef numpy.ndarray[double, ndim = 1] costs       = numpy.empty(len(model.actions))
    cdef numpy.ndarray[double, ndim = 2] utilities   = numpy.empty(history.shape)
    cdef numpy.ndarray[unsigned int, ndim = 1] sizes = numpy.empty(len(model.actions), numpy.uint)

    for (i, action) in enumerate(model.actions):
        costs[i] = action.cost
        sizes[i] = len(action.outcomes)

        for (j, outcome) in enumerate(action.outcomes):
            utilities[i, j] = outcome.utility

    # set up our fixed-size stack
    cdef BellmanStackEntry* stack = <BellmanStackEntry*>malloc(horizon * sizeof(BellmanStackEntry))

    for i in xrange(horizon):
        stack[i].predictions = <double*>malloc(history.size * sizeof(double))

    # invoke the inner loop
    cdef numpy.ndarray[unsigned int, ndim = 2, mode = "c"] fancy_history = history

    (utility, plan) = \
        compute_bellman_utility_inner(
            model.predictor,
            costs,
            utilities,
            sizes,
            horizon,
            budget,
            discount,
            <unsigned int*>fancy_history.data,
            stack,
            )

    return (utility, [model.actions[i] for i in plan])

cdef compute_bellman_utility_inner(
    Predictor predictor,
    py_costs,
    py_utilities,
    py_sizes,
    unsigned int horizon,
    double budget,
    double discount,
    unsigned int* history,
    BellmanStackEntry* entry,
    ):
    """
    Compute the expected utility of a state.
    """

    if horizon == 0:
        return (0.0, [])

    # interpret parameters
    cdef numpy.ndarray[double, ndim = 1] costs       = py_costs
    cdef numpy.ndarray[double, ndim = 2] utilities   = py_utilities
    cdef numpy.ndarray[unsigned int, ndim = 1] sizes = py_sizes

    cdef size_t M = py_utilities.shape[0]
    cdef size_t D = py_utilities.shape[1]

    # prediction outcomes at this state
    cdef double* predictions = entry.predictions

    predictor.predict_raw(history, predictions)

    # used by the loop
    cdef double e
    cdef double best_e = -1.0
    cdef Py_ssize_t best_action = 0

    cdef double action_cost
    cdef Py_ssize_t i
    cdef Py_ssize_t j

    best_plan = None

    for i in xrange(costs.shape[0]):
        action_cost = costs[i]

        if action_cost <= budget:
            e = 0.0

            for j in xrange(sizes[i]):
                outcome_utility = utilities[i, j]

                if outcome_utility > 0.0:
                    e += predictions[i * D + j] * discount**action_cost * outcome_utility
                else:
                    history[i * D + j] += 1

                    (t_e, t_plan) = \
                        compute_bellman_utility_inner(
                            predictor,
                            costs,
                            utilities,
                            sizes,
                            horizon - 1,
                            budget - action_cost,
                            discount,
                            history,
                            entry + 1,
                            )

                    history[i * D + j] -= 1

                    e += predictions[i * D + j] * discount**action_cost * t_e

            if e >= best_e:
                best_e      = e
                best_action = i
                best_plan   = t_plan

    if horizon >= 2:
        log.info(
            "%i: %f %s",
            horizon,
            best_e,
            None if best_e < 0.0 else best_action,
            )

    if best_e < 0.0:
        return (0.0, [])
    else:
        return (best_e, [best_action] + best_plan)

