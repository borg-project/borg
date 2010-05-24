"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import abstractproperty
from cargo.sugar import ABC

class Action(ABC):
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

class Outcome(ABC):
    """
    An outcome of an action in the world.
    """

    @abstractproperty
    def utility(self):
        """
        The utility of this outcome.
        """

