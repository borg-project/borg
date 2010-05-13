"""
utexas/portfolio/world.py

Actions, tasks, outcomes, and other pieces of the world.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import abstractproperty
from cargo.sugar import ABC

# class Task(ABC):
#     """
#     A task in the world.
#     """

#     def __init__(self, ntask, path):
#         """
#         Initialize.
#         """

#         self.world = world
#         self.ntask = ntask
#         self.path = path

#     def sample_action(self, action):
#         return self.world.sample_action(self, action)

class Action(ABC):
    """
    An action in the world.
    """

    @property
    def description(self):
        """
        A human-readable description of this action.
        """

    @abstractproperty
    def cost(self):
        """
        The typical cost of taking this action.
        """

class Outcome(object):
    """
    An outcome of an action in the world.
    """

    pass

