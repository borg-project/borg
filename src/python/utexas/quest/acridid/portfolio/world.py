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

class Task(ABC):
    """
    A task in the world.
    """

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

class Action(object):
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

# class Actions(Sequence):
#     """
#     Actions in the world.
#     """

#     @abstractmethod
#     def get_by(self, nsolver, cutoff):
#         """
#         Get an action by solver index and cutoff.
#         """

#         pass

class Outcome(object):
    """
    An outcome of an action in the world.
    """

    # properties
#     utility = property(lambda self: self.world.utilities[self.n])

# FIXME should just be a tuple
# class Outcomes(Sequence):
#     """
#     Outcomes of actions in the world.
#     """

#     pass

class World(ABC):
    """
    A description of the environment.
    """

    def counts_from_events(self, events):
        """
        Return a counts matrix from an events dictionary.
        """

        counts = numpy.zeros((self.ntasks, self.nactions, self.noutcomes), numpy.uint)

        for (task, pairs) in events.iteritems():
            for (action, outcome) in pairs:
                counts[task.n, action.n, outcome.n] += 1

        return counts

    def act_all(self, tasks, actions, nrestarts = 1, random = numpy.random):
        """
        Return a history of C{nrestarts} outcomes sampled from each of C{tasks}.

        @return: {task: (action, outcome)}
        """

        return dict((t, [(a, self.act(t, a, random)) for a in actions]) for t in tasks)

    @abstractmethod
    def act(self, task, action, random = numpy.random):
        """
        Retrieve a random sample.
        """

        pass

    # properties
    ntasks = property(lambda self: len(self.tasks))
    nactions = property(lambda self: len(self.actions))
    noutcomes = property(lambda self: len(self.outcomes))

