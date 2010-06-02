"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from itertools               import izip
from borg.portfolio.planners import AbstractPlanner

def compute_bellman_utility(model, horizon, budget, history):
    """
    Compute the expected utility of a state.
    """

    if horizon == 0 or budget <= 0:
        return (0.0, [])
    else:
        best_e      = 0.0
        best_action = None
        best_plan   = None
        predictions = model.predict(history, None)

        for action in model._actions:
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
                            history + [(action, o)],
                            )

                    e += p * t_e

            if e >= best_e:
                best_e      = e
                best_action = action
                best_plan   = t_plan

        print horizon, [(a.description, b.utility) for (a, b) in history], best_e, best_action.description

        return (best_e, [best_action] + best_plan)

def compute_bellman_plan(model, horizon, budget):
    """
    Compute the Bellman-optimal plan.
    """

    actions = model._actions

    (_, plan) = compute_bellman_utility([], 0, 0.0, [])

    print "computed plan follows"

    for action in plan:
        print action.description

class BellmanPlanner(AbstractPlanner):
    """
    An optimal, exponential-time, finite-horizon, cost-limited planner.
    """

    def __init__(self, model, horizon, budget):
        """
        Initialize.
        """

        compute_bellman_plan(model, horizon, budget)

    def select(self, predicted, actions, random):
        """
        Select an action given the probabilities of outcomes.
        """

        raise NotImplementedError()

    @staticmethod
    def build(request, trainer, model):
        """
        Build a sequence strategy as requested.
        """

        return BellmanPlanner(model, request["horizon"], request["budget"])

