"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import  numpy
cimport numpy

from cargo.log import get_logger

log = get_logger(__name__)

def compute_bellman_utility(
    model,
    unsigned int horizon,
    double budget,
    double discount,
    numpy.ndarray[unsigned int, ndim = 2] history,
    ):
    """
    Compute the expected utility of a state.
    """

    cdef double e

    if horizon == 0:
        return (0.0, [])
    else:
        best_e      = 0.0
        best_action = None
        best_plan   = None
        predictions = model.predict(history, None)

        for (i, action) in enumerate(model.actions):
            if action.cost <= budget:
                e = 0.0

                for (j, o) in enumerate(action.outcomes):
                    p = predictions[i, j]

                    if o.utility > 0.0:
                        e += p * discount**action.cost * o.utility
                    else:
                        history[i, j] += 1

                        (t_e, t_plan) = \
                            compute_bellman_utility(
                                model,
                                horizon - 1,
                                budget - action.cost,
                                discount,
                                history,
                                )

                        history[i, j] -= 1

                        e += p * discount**action.cost * t_e

                if e >= best_e:
                    best_e      = e
                    best_action = action
                    best_plan   = t_plan

        if horizon >= 2:
            log.info(
                "%i: %f %s",
                horizon,
                best_e,
                None if best_action is None else best_action.description,
                )

        if best_action is None:
            return (0.0, [])
        else:
            return (best_e, [best_action] + best_plan)

