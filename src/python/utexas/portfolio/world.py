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

class World(ABC):
    """
    A description of the environment.
    """

    def counts_from_events(self, events):
        """
        Return a counts matrix from an events dictionary.
        """

        counts = numpy.zeros((self.ntasks, self.nactions, self.noutcomes), numpy.uint)

        for (task, action, outcome) in events:
            counts[task.n, action.n, outcome.n] += 1

        return counts

    def act_all(self, tasks, actions, nrestarts = 1, random = numpy.random):
        """
        Return a history of C{nrestarts} outcomes sampled from each of C{tasks}.

        @return: [(task, action, outcome)]
        """

        events = []

        for (task, action) in product(tasks, actions):
            events.extend((task, action, o) for o in self.act(task, action, nrestarts, random))

        return events

    @abstractmethod
    def act(self, task, action, nrestarts = 1, random = numpy.random):
        """
        Sample action outcomes.
        """

        pass

    def act_once(self, task, action, random = numpy.random):
        """
        Sample an action outcome.
        """

        (outcome,) = self.act(task, action, random = random)

        return outcome

    # properties
    ntasks    = property(lambda self: len(self.tasks))
    nactions  = property(lambda self: len(self.actions))
    noutcomes = property(lambda self: len(self.outcomes))

