"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import abstractmethod
from cargo.log   import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

def build_strategy(request, trainer):
    """
    Build a selection strategy as requested.
    """

    builders = {
        "sequence" : SequenceStrategy.build,
        "fixed"    : FixedStrategy.build,
        "modeling" : ModelingStrategy.build,
        "bellman"  : BellmanStrategy.build,
        }

    return builders[request["type"]](request, trainer)

class AbstractStrategy(ABC):
    """
    Abstract base for selection strategies.
    """

    @abstractmethod
    def select(self, budget, random):
        """
        A generator that yields actions and receives (outcome, next_budget).
        """

class SequenceStrategy(AbstractStrategy):
    """
    A strategy the follows an iterable sequence for every task.
    """

    def __init__(self, actions):
        """
        Initialize.
        """

        self.actions = actions

    def select(self, budget, random):
        """
        A generator that yields actions and receives (outcome, next_budget).
        """

        for selected in self.actions:
            (_, budget) = yield selected

        while True:
            yield None

    @staticmethod
    def build(request, trainer):
        """
        Build a sequence strategy as requested.
        """

        raise NotImplementedError()

class FixedStrategy(SequenceStrategy):
    """
    A strategy that repeats a fixed action.
    """

    def __init__(self, action):
        """
        Initialize.
        """

        from itertools import repeat

        SequenceStrategy.__init__(self, repeat(action))

    @staticmethod
    def build(request, trainer):
        """
        Build a fixed strategy as requested.
        """

        raise NotImplementedError()

class ModelingStrategy(AbstractStrategy):
    """
    A strategy that employs a model of its actions.
    """

    def __init__(self, model, planner):
        """
        Initialize.
        """

        self.model   = model
        self.planner = planner
        self._action_indices = dict((a, i) for (i, a) in enumerate(self.model.actions))

    def select(self, budget, random):
        """
        Select an action, yield it, and receive its outcome.
        """

        import numpy

        dimensions = (len(self.model.actions), max(len(a.outcomes) for a in self.model.actions))
        history    = numpy.zeros(dimensions, numpy.uint)

        while True:
            # predict, then make a selection
            predicted         = self.model.predict(history, random)
            selected          = self.planner.select(predicted, budget, random)
            (outcome, budget) = yield selected

            history[self.model.actions.index(selected), selected.outcomes.index(outcome)] += 1

    @staticmethod
    def build(request, trainer):
        """
        Build a modeling selection strategy as requested.
        """

        from borg.portfolio.models   import build_model
        from borg.portfolio.planners import build_planner

        model   = build_model(request["model"], trainer)
        planner = build_planner(request["planner"], trainer, model)

        return ModelingStrategy(model, planner)

class BellmanStrategy(SequenceStrategy):
    """
    A strategy that employs a model of its actions.
    """

    def __init__(self, model, horizon, budget, discount = 1.0):
        """
        Initialize.
        """

        from cargo.temporal                import TimeDelta
        from borg.portfolio.bellman        import compute_bellman_plan
        from borg.portfolio.decision_world import DecisionWorldAction

        plan = compute_bellman_plan(model, horizon, budget, discount)

        plan[-1] = DecisionWorldAction(plan[-1].solver, TimeDelta(seconds = 1e6))

        log.info("Bellman plan follows (horizon %i, budget %f)", horizon, budget)

        for (i, action) in enumerate(plan):
            log.info("action %i: %s", i, action.description)

        SequenceStrategy.__init__(self, plan)

    @staticmethod
    def build(request, trainer):
        """
        Build a modeling selection strategy as requested.
        """

        from borg.portfolio.models import build_model

        model = build_model(request["model"], trainer)

        return BellmanStrategy(model, request["horizon"], request["budget"], request["discount"])

