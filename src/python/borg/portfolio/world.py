"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc                  import (
    abstractmethod,
    abstractproperty,
    )
from cargo.sugar          import ABC
from borg.portfolio._base import (
    Action,
    Outcome,
    )

def build_trainer(domain, task_uuids, Session, extrapolation = 1):
    """
    Build a trainer as requested.
    """

    from borg.portfolio.decision_world import DecisionTrainer

    trainer_builders = {
        "sat" : DecisionTrainer,
        "pb"  : DecisionTrainer,
        }

    return trainer_builders[domain](task_uuids, Session, extrapolation = extrapolation)

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

