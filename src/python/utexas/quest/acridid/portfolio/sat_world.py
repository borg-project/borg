"""
utexas/acridid/portfolio/sat_world.py

The world of SAT.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from itertools import izip
from collections import (
    Sequence,
    defaultdict,
    )
from cargo.log import DefaultLogger
from cargo.flags import (
    Flag,
    FlagSet,
    IntRanges,
    )

log = get_logger(__name__)

class SAT_WorldAction(Action):
    """
    An action in the world.
    """

    def __init__(self, n, solver, configuration, cutoff):
        """
        Initialize.
        """

        self.n = n
        self.solver        = solver
        self.configuration = configuration
        self.cutoff        = cutoff

    def __str__(self):
        return "%s_%ims" % (self.solver_name, int(self.cutoff * 1000))

class SAT_WorldTask(object):
    """
    A task in the world.
    """

    def __init__(self, n, task):
        """
        Initialize.

        @param task: SAT task description.
        """

        self.n = n
        self.task = task

class SAT_Outcome(object):
    """
    An outcome of an action in the world.
    """

    def __init__(self, n, utility):
        """
        Initialize.
        """

        self.n       = n
        self.utility = utility

    def __str__(self):
        """
        Return a human-readable description of this outcome.
        """

        return str(self.utility)

    # properties
    # FIXME
    utility = property(lambda self: self.world.utilities[self.n])

    # constants
    SAT     = SAT_Outcome(0, 1.0)
    UNSAT   = SAT_Outcome(0, 1.0)
    UNKNOWN = SAT_Outcome(1, 0.0)

class SAT_World(object):
    """
    Components of the SAT world.
    """

    def __init__(self, actions, tasks):
        """
        Initialize.
        """

        self.actions   = actions
        self.tasks     = tasks
        self.utilities = numpy.array([1.0, 0.0])
        self.outcomes  = (SAT_Outcome.SAT, SAT_Outcome.UNSAT)

    def sample_action(self, task, action):
        """
        Retrieve a random sample.
        """

        # FIXME
        return self.samples.sample_action(task, action)

    # properties
    ntasks = property(lambda self: len(self.tasks))
    nactions = property(lambda self: len(self.actions))
    noutcomes = property(lambda self: len(self.outcomes))

