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

class SAT_Action(Action):
    """
    An action in the world.
    """

    def __init__(self, n, nsolver, solver_name, cutoff):
        """
        Initialize.
        """

        self.n = n
        self.solver = solver
        self.cutoff = cutoff

    def __str__(self):
        return "%s_%ims" % (self.solver_name, int(self.cutoff * 1000))

class SAT_WorldTask(object):
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

class SAT_Outcome(object):
    """
    An outcome of an action in the world.
    """

    def __init__(self, world, n):
        """
        Initialize.
        """

        self.world = world
        self.n = n

    def __str__(self):
        """
        Return a human-readable description of this outcome.
        """

        return str(self.utility)

    # properties
    utility = property(lambda self: self.world.utilities[self.n])
    true_utility = property(__get_true_utility)

class WorldDescriptionActions(Sequence):
    """
    Actions in the world.
    """

    def __init__(self, world, exclude_nnsolver = []):
        """
        Initialize.
        """

        self.world = world
        self.supplies = world.supplies
        self.cutoffs = world.cutoffs
        self.solvers_node = self.supplies.solvers_file.getNode("/solvers")
        self.actions = []
        self.actions_by = {}

        for solver_row in self.solvers_node:
            if solver_row.nrow in exclude_nnsolver:
                continue

            for cutoff in self.cutoffs:
                action = WorldAction(len(self.actions), solver_row.nrow, solver_row["name"], cutoff)

                self.actions.append(action)
                self.actions_by[(action.nsolver, action.cutoff)] = action

    def __len__(self):
        """
        Return the number of actions.
        """

        return len(self.actions)

    def __getitem__(self, naction):
        """
        Return a specific action.
        """

        return self.actions[naction]

    def get_by(self, nsolver, cutoff):
        return self.actions_by[(nsolver, cutoff)]

class WorldDescriptionTasks(Sequence):
    """
    Tasks in the world.
    """

    def __init__(self, world, nntasks = None):
        """
        Initialize.
        """

        self.world = world
        self.supplies = world.supplies
        self.tasks_table = self.supplies.tasks_file.getNode("/problems")

        if nntasks is None:
            def yield_tasks():
                for task_row in self.tasks_table:
                    yield WorldTask(self.world, task_row.nrow, task_row.nrow, task_row["path"])
        else:
            def yield_tasks():
                for (n, ntask) in enumerate(nntasks):
                    task_record = self.tasks_table[ntask]
                    
                    yield WorldTask(self.world, n, ntask, task_record["path"])

        self.tasks_list = list(yield_tasks())

    def __len__(self):
        """
        Return the number of tasks.
        """

        return len(self.tasks_list)

    def __getitem__(self, ntask):
        """
        Return a specific task.
        """

        if ntask >= len(self):
            raise IndexError()

        return self.tasks_list[ntask]

class SAT_Samples(object):
    """
    The solver outcome samples.
    """

    def __init__(self, world):
        """
        Initialize.
        """

        # parameters
        self.world = world
        self.supplies = world.supplies
        self.samples_table = self.supplies.samples_file.getNode("/samples")

    def get_samples_list(self, task, action):
        """
        Get all task-action outcome samples stored.
        """

        key = (action.nsolver, action.cutoff, task.ntask)

        return self.indexed_samples[key]

    def get_outcomes(self, task, action):
        """
        Get all task-action outcome samples stored.
        """

        for sample in self.get_samples_list(task, action):
            yield Outcome.of_SAT(self.world, sample)

    def sample_action(self, task, action):
        """
        Retrieve a random sample.
        """

        samples = self.get_samples_list(task, action)

        if samples:
            sample = samples[numpy.random.randint(0, len(samples))]

            return Outcome.of_SAT(self.world, sample)
        else:
            raise RuntimeError("no samples of %s on task %i" % (str(action), task.ntask))

class SAT_World(object):
    """
    Components of the SAT world.
    """

    def __init__(self, cutoffs, actions, tasks):
        """
        Initialize.
        """

        self.cutoffs   = FIXME
        self.actions   = WorldDescriptionActions(self, exclude_nnsolver = [9])
        self.tasks     = WorldDescriptionTasks(self, self.flags.nntasks_world)
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

