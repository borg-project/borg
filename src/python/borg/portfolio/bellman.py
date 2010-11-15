"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log import get_logger

log = get_logger(__name__)

class BellmanCore(object):
    """
    Compute a Bellman-optimal plan.
    """

    def __init__(self, model, discount, enabled = None):
        """
        Initialize.
        """

        if enabled is None:
            enabled = [True] * len(model.actions)

        self._model     = model
        self._discount  = discount
        self._enabled   = enabled

    def plan_from_start(self, horizon, budget):
        """
        Plan from the blank-history state.
        """

        import numpy

        dimensions = (len(self._model.actions), max(len(a.outcomes) for a in self._model.actions))
        history    = numpy.zeros(dimensions, numpy.uint)

        return self.plan(horizon, budget, history)

    def plan(self, horizon, budget, history):
        """
        Compute an optimal plan.
        """

        predictions   = self._model.predict(history, None)
        best_expected = 0.0
        best_plan     = []

        for i in xrange(len(self._model.actions)):
            action = self._model.actions[i]

            # action is disabled?
            if not self._enabled[i]:
                this_expected = 0.0
            # recurse into later states?
            elif horizon > 1:
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
                            self.plan(
                                horizon - 1,
                                budget - action.cost,
                                history,
                                )

                        history[i, j] -= 1

                    # update the expectation for this action
                    reward         = outcome.utility + self._discount * inner_utility
                    this_expected += predictions[i, j] * reward
            # we're at a base case?
            elif budget >= action.cost:
                this_expected = sum(predictions[i, j] * o.utility for (j, o) in enumerate(action.outcomes))
                inner_plan    = []
            # we can't afford this action
            else:
                this_expected = 0.0

            # update the max over actions
            if this_expected > best_expected:
                best_expected = this_expected
                best_plan     = [action] + inner_plan

        return (best_expected, best_plan)

