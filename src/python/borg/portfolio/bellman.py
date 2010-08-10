"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log import get_logger

log = get_logger(__name__)

def compute_bellman_plan(model, horizon, budget, discount, history = None):
    """
    Compute an optimal plan.
    """

    # parameters
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if history is None:
        import numpy

        dimensions = (len(model.actions), max(len(a.outcomes) for a in model.actions))
        history    = numpy.zeros(dimensions, numpy.uint)

    # compute the plan
    return _compute_bellman_plan(model, horizon, budget, discount, history)

def _compute_bellman_plan(model, horizon, budget, discount, history):
    """
    Compute an optimal plan.
    """

    predictions   = model.predict(history, None)
    best_expected = 0.0
    best_plan     = []

    for i in xrange(len(model.actions)):
        action = model.actions[i]

        if horizon > 1:
            # we'll recurse into later states
            this_expected = 0.0

            for j in xrange(len(action.outcomes)):
                outcome = action.outcomes[j]

                if outcome.utility > 0.0:
                    # halting state
                    inner_utility = 0.0
                    inner_plan    = []
                else:
                    # compute the utility of later actions
                    history[i, j] += 1

                    (inner_utility, inner_plan) = \
                        _compute_bellman_plan(
                            model,
                            horizon - 1,
                            budget - action.cost,
                            discount,
                            history,
                            )

                    history[i, j] -= 1

                # update the expectation for this action
                this_expected += predictions[i, j] * (outcome.utility + discount * inner_utility)
        elif budget > action.cost:
            # we're at a base case
            this_expected = sum(predictions[i, j] * o.utility for (j, o) in enumerate(action.outcomes))
            inner_plan    = []
        else:
            # we can't afford this action
            this_expected = 0.0

        # update the max over actions
        if this_expected > best_expected:
            best_expected = this_expected
            best_plan     = [action] + inner_plan

    return (best_expected, best_plan)

