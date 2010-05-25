"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import (
    abstractmethod,
    abstractproperty,
    )
from cargo.sugar import ABC

def build_trainer(domain, task_uuids, Session):
    """
    Build a trainer as requested.
    """

    from borg.portfolio.decision_world import DecisionTrainer

    trainer_builders = {
        "sat" : DecisionTrainer,
        "pb"  : DecisionTrainer,
        }

    return trainer_builders[domain](task_uuids, Session)

class AbstractTrainer(ABC):
    """
    Grant a portfolio access to training data.
    """

    @abstractmethod
    def build_actions(request):
        """
        Build a list of actions from a configuration request.
        """

    @abstractmethod
    def get_data(self, action):
        """
        Provide per-task {outcome: count} maps to the trainee.
        """

class AbstractAction(ABC):
    """
    An action in the world.
    """

    @property
    def description(self):
        """
        A human-readable description of this action.
        """

        raise NotImplementedError()

    @abstractproperty
    def cost(self):
        """
        The typical cost of taking this action.
        """

    @abstractproperty
    def outcomes(self):
        """
        The possible outcomes of this action.
        """

class AbstractOutcome(ABC):
    """
    An outcome of an action in the world.
    """

    @abstractproperty
    def utility(self):
        """
        The utility of this outcome.
        """

