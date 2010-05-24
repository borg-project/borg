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
        }

    return builders[request["type"]](request, trainer)

class AbstractStrategy(ABC):
    """
    Abstract base for selection strategies.
    """

    @abstractmethod
    def select(self, task, budget, random):
        """
        Select an action, yield it, and receive its outcome.
        """

class SequenceStrategy(AbstractStrategy):
    """
    A strategy the follows an iterable sequence.
    """

    def __init__(self, actions):
        """
        Initialize.
        """

        self.action_sequence = iter(actions)

    def select(self, task, budget, random):
        """
        Select an action, yield it, and receive its outcome.
        """

        selected = self.action_sequence.next()

        if selected.cost > budget:
            yield None
        else:
            yield selected

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

        SequenceSelectionStrategy.__init__(self, repeat(action))

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
        self.history = []

    def select(self, task, budget, random):
        """
        Select an action, yield it, and receive its outcome.
        """

        # predict, then make a selection
        predicted = self.model.predict(task, self.history, random)
        selected  = self.planner.select(predicted, budget, random)
        outcome   = yield selected

        # remember its result
        if outcome is not None:
            self.history.append((task, selected, outcome))

    @staticmethod
    def build(request, trainer):
        """
        Build a modeling selection strategy as requested.
        """

        from borg.portfolio.models   import build_model
        from borg.portfolio.planners import build_planner

        return \
            ModelingStrategy(
                build_model(request["model"], trainer),
                build_planner(request["planner"], trainer),
                )

