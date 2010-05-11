"""
utexas/portfolio/world.py

Actions, tasks, outcomes, and other pieces of the world.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc         import abstractmethod
from itertools   import product
from cargo.log   import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

# FIXME these are all out of date

class Task(ABC):
    """
    A task in the world.
    """

    def __init__(self, ntask, path):
        """
        Initialize.
        """

        self.world = world
        self.ntask = ntask
        self.path = path

    def sample_action(self, action):
        return self.world.sample_action(self, action)

class Action(object):
    """
    An action in the world.
    """

    def __init__(self, n, nsolver, solver_name, cost):
        """
        Initialize.
        """

        self.n           = n
        self.nsolver     = nsolver
        self.solver_name = solver_name
        self.cost      = cost

    def __str__(self):
        return "%s_%ims" % (self.solver_name, int(self.cutoff * 1000))

class Outcome(object):
    """
    An outcome of an action in the world.
    """

    pass

def get_positive_counts(counts):
    """
    Return only rows with recorded outcomes.
    """

    return counts[numpy.sum(numpy.sum(counts, 1), 1) > 0]

