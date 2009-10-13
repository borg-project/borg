"""
utexas/acridid/portfolio/world.py

Actions, tasks, outcomes, and other pieces of the world.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc import abstractmethod
from collections import (
    Sequence,
    defaultdict,
    )
from cargo.log import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

# general strategy: implement via script construction first; persist later

class SAT_WorldTask(ABC):
    """
    A task in the world.
    """

    __tablename__ = "sat_world_tasks"

#     world = 

    # FIXME should be constructed from a SAT task during world construction
    # FIXME (in the particular evaluation script in Chanal)

    def __init__(self, world, n, ntask, path):
        """
        Initialize.
        """

        self.world = world
        self.n = n
        self.ntask = ntask
        self.path = path

    def sample_action(self, action):
        return self.world.sample_action(self, action)

# FIXME should just be a tuple
# class Tasks(Sequence):
#     """
#     Tasks in the world.
#     """

#     pass

class SAT_WorldAction(object):
    """
    An action in the world.
    """

    def __init__(self, n, nsolver, solver_name, cutoff):
        """
        Initialize.
        """

        self.n = n
        self.nsolver = nsolver
        self.solver_name = solver_name
        self.cutoff = cutoff

    def __str__(self):
        return "%s_%ims" % (self.solver_name, int(self.cutoff * 1000))

class Actions(Sequence):
    """
    Actions in the world.
    """

    @abstractmethod
    def get_by(self, nsolver, cutoff):
        """
        Get an action by solver index and cutoff.
        """

        pass

# FIXME not persisted
class Outcome(object):
    """
    An outcome of an action in the world.
    """

    @staticmethod
    def of_SAT(world, success):
        """
        Construct an outcome in the SAT domain.
        """

        if success:
            return Outcome(world, 0)
        else:
            return Outcome(world, 1)

    # properties
    utility = property(lambda self: self.world.utilities[self.n])

# FIXME should just be a tuple
class Outcomes(Sequence):
    """
    Outcomes of actions in the world.
    """

    pass

class World(ABC):
    """
    A description of the environment.
    """

    @abstractmethod
    def sample_action(self, task, action):
        """
        Retrieve a random sample.
        """

        pass

    # properties
    ntasks = property(lambda self: len(self.tasks))
    nactions = property(lambda self: len(self.actions))
    noutcomes = property(lambda self: len(self.outcomes))

