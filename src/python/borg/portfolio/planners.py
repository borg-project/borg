"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc import (
    ABCMeta,
    abstractmethod,
    )
from cargo.log import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

def build_planner(request, trainer, model):
    """
    Build an action planner as requested.
    """

    builders = {
        "hard_myopic" : HardMyopicPlanner.build,
        "soft_myopic" : SoftMyopicPlanner.build,
        "bellman"     : BellmanPlanner.build,
        }

    return builders[request["type"]](request, trainer, model)

class AbstractPlanner(ABC):
    """
    Interface for action selection schemes.
    """

    @abstractmethod
    def select(self, predicted, budget, random):
        """
        Select an action given the probabilities of outcomes.
        """

class HardMyopicPlanner(AbstractPlanner):
    """
    Deterministic greedy action selection.
    """

    def __init__(self, discount):
        """
        Initialize.
        """

        self.discount = discount

    def select(self, predicted, budget, random):
        """
        Select an action given the probabilities of outcomes.
        """

        feasible = [(a, ps) for (a, ps) in predicted.iteritems() if a.cost <= budget]

        if feasible:
            (selected, _) = \
                max(
                    feasible,
                    key = lambda (a, ps): sum(p*o.utility*self.discount**a.cost for (p, o) in zip(ps, a.outcomes)),
                    )

            return selected
        else:
            return None

    @staticmethod
    def build(request, trainer, model):
        """
        Build a sequence strategy as requested.
        """

        return HardMyopicPlanner(request["discount"])

class SoftMyopicPlanner(AbstractPlanner):
    """
    Probabilistic greedy action selection.
    """

    def __init__(self, discount, temperature = 1.0):
        """
        Initialize.
        """

        self.discount    = discount
        self.temperature = temperature

    def select(self, predicted, actions, random):
        """
        Select an action given the probabilities of outcomes.
        """

        # convert to expectation
        import numpy

        expected       = numpy.sum(predicted * self.world.utilities, 1)
        discounted     = numpy.fromiter((expected[a.n]*(self.discount**a.cutoff.as_s) for a in actions), numpy.double)
        probabilities  = numpy.exp(discounted / self.temperature)
        probabilities /= numpy.sum(probabilities)
        ((naction,),)  = numpy.nonzero(random.multinomial(1, probabilities))
        action         = actions[naction]

        log.detail("probabilities: %s", probabilities)
        log.detail("selected action %i (p = %.4f): %s", naction, probabilities[naction], action)

        return action

    @staticmethod
    def build(request, trainer, model):
        """
        Build a sequence strategy as requested.
        """

        return SoftMyopicPlanner(request["discount"], request["temperature"])

class BellmanPlanner(AbstractPlanner):
    """
    An optimal, exponential-time finite-horizon cost-limited planner.
    """

    def __init__(self, model, horizon, budget):
        """
        Initialize.
        """

        from itertools import izip

        actions = model._actions

        def eu(history, depth, cost, plan):
            """
            Compute the expected utility of a state.
            """

            if depth == horizon or cost >= budget:
                return (0.0, plan)
            else:
                best_e      = 0.0
                best_action = None
                predictions = model.predict(None, history, None)

                for action in actions:
                    e = 0.0

                    for (o, p) in izip(action.outcomes, predictions[action]):
                        if o.utility > 0.0:
                            e += p * o.utility
                        else:
                            (t_e, t_plan) = eu(history + [(None, action, o)], depth + 1, cost + action.cost, plan)
                            e += p * t_e

                    if e >= best_e:
                        best_e      = e
                        best_action = action

                    if depth <= 0:
                        print depth, action.description, e

                return (best_e, [best_action] + t_plan)

        (_, plan) = eu([], 0, 0.0, [])

        print "computed plan follows"

        for action in plan:
            print action.description

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

