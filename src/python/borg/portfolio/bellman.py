"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from itertools               import izip
from cargo.log               import get_logger
from borg.portfolio.planners import AbstractPlanner

log = get_logger(__name__)

def compute_bellman_utility(model, horizon, budget, discount, history):
    """
    Compute the expected utility of a state.
    """

    if horizon == 0:
        return (0.0, [])
    else:
        best_e      = 0.0
        best_action = None
        best_plan   = None
        predictions = model.predict(history, None)

        for action in model.actions:
            if action.cost <= budget:
                e = 0.0

                for (o, p) in izip(action.outcomes, predictions[action]):
                    if o.utility > 0.0:
                        e += p * o.utility
                    else:
                        (t_e, t_plan) = \
                            compute_bellman_utility(
                                model,
                                horizon - 1,
                                budget - action.cost,
                                discount,
                                history + [(action, o)],
                                )

                        e += p * discount * t_e

                if e >= best_e:
                    best_e      = e
                    best_action = action
                    best_plan   = t_plan

        if horizon >= 2:
            log.info(
                "%i: %s %f %s",
                horizon,
                str([a.description for (a, _) in history]),
                best_e,
                None if best_action is None else best_action.description,
                )

        if best_action is None:
            return (0.0, [])
        else:
            return (best_e, [best_action] + best_plan)

def compute_bellman_plan(model, horizon, budget, discount = 1.0):
    """
    Compute the Bellman-optimal plan.
    """

    (_, plan) = compute_bellman_utility(model, horizon, budget, discount, [])

    return plan

