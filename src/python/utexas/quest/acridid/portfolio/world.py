"""
utexas/acridid/portfolio/world.py

Actions, tasks, outcomes, and other pieces of the world.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc import (
    ABCMeta,
    abstractmethod,
    )
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

def get_sample_key(leaf, i):
    """
    Return the row key for a sample table row.
    """

    row = leaf[i]

    return (row[_solver_col_index], row[_cutoff_col_index], row[_problem_col_index])

def get_sample_row_key(row):
    """
    Return the row key for a sample table row.
    """

    return (row[_solver_col_index], row[_cutoff_col_index], row[_problem_col_index])

class WorldAction(object):
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

class WorldTask(object):
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

class Outcome(object):
    """
    An outcome of an action in the world.
    """

    def __init__(self, world, n):
        """
        Initialize.
        """

        self.world = world
        self.n = n
        self.__true_utility = None

    def __str__(self):
        """
        Return a human-readable description of this outcome.
        """

        return str(self.utility)

    def __get_true_utility(self):
        """
        Return the true utility, or utility, if no true utility.
        """

        if self.__true_utility is None:
            return self.utility
        else:
            return self.__true_utility

    def with_true_utility(self, true_utility):
        """
        Return a copy of this outcome with differing true utility.
        """

        outcome = Outcome(self.world, self.n)
        outcome.__true_utility = true_utility

        return outcome

    @staticmethod
    def of_SAT(world, success):
        """
        Construct an outcome in the SAT domain.
        """

        if success:
            return Outcome(world, 0)
        else:
            return Outcome(world, 1)

    @staticmethod
    def of_MAX_SAT(world, optimum, total_weight):
        """
        Construct an outcome in the MAX-SAT domain.
        """

        assert 0 <= optimum <= total_weight

        (h, _) = numpy.histogram(optimum, world.noutcomes, (0.0, total_weight), new = True)
        ((n,),) = numpy.nonzero(h)

        return world.outcomes[n]

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

class Outcomes(Sequence):
    def __init__(self, world, noutcomes):
        """
        Initialize.
        """

        self.world = world
        self.noutcomes = noutcomes

    def __len__(self):
        return self.noutcomes

    def __getitem__(self, noutcome):
        if noutcome >= len(self):
            raise IndexError()

        return Outcome(self.world, noutcome)

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

        if self.supplies.samples_dict is None:
            log.info("indexing samples in memory")

            self.indexed_samples = self._build_samples_dictionary()
        else:
            self.indexed_samples = self.supplies.samples_dict

        log.info("samples index has %i entries", len(self.indexed_samples))

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

    def _build_samples_dictionary(self):
        """
        Build the dictionary of samples.
        """

        samples = self.samples_table.read()
        indexed_samples = defaultdict(list)
        solver_col_index = _solver_col_index
        cutoff_col_index = _cutoff_col_index
        problem_col_index = _problem_col_index
        success_col_index = _success_col_index

        for row in samples:
            key = (row[solver_col_index], row[cutoff_col_index], row[problem_col_index])

            indexed_samples[key].append(row[success_col_index])

        return indexed_samples

class MAX_SAT_Samples(object):
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
        self.runs_table = self.supplies.samples_file.getNode("/samples/runs")
        self.times_vlarray = self.supplies.samples_file.getNode("/samples/times")
        self.optima_vlarray = self.supplies.samples_file.getNode("/samples/optima")

        if self.supplies.samples_dict is None:
            log.info("indexing samples in memory")

            self.indexed_samples = self._build_samples_dictionary()
        else:
            self.indexed_samples = self.supplies.samples_dict

        log.info("samples index has %i entries", len(self.indexed_samples))

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

        for (noutcome, true_utility) in self.get_samples_list(task, action):
            yield self.world.outcomes[noutcome].with_true_utility(true_utility)

    def sample_action(self, task, action):
        """
        Retrieve a random sample.
        """

        nnoutcome = self.get_samples_list(task, action)
        (noutcome, true_utility) = nnoutcome[numpy.random.randint(0, len(nnoutcome))]

        return self.world.outcomes[noutcome].with_true_utility(true_utility)

    def _build_samples_dictionary(self):
        """
        Build the dictionary of samples.
        """

        indexed_samples = defaultdict(list)
        problems_table = self.supplies.tasks_file.getNode("/problems")

        for (nrun, run_record) in enumerate(self.runs_table.read()):
            task_record = problems_table[run_record["problem"]]
            times_optima = \
                izip(
                    reversed(self.times_vlarray[nrun]),
                    reversed(self.optima_vlarray[nrun]),
                    )
            time = None

            for cutoff in reversed(self.world.cutoffs):
                key = (run_record["solver"], cutoff, run_record["problem"])

                while time is None or time > cutoff:
                    (time, optimum) = times_optima.next()

                outcome = Outcome.of_MAX_SAT(self.world, optimum, task_record["clauses"])

                indexed_samples[key].append((outcome.n, optimum))

        return indexed_samples

class WorldDescription(object):
    """
    A description of the environment.
    """

    class Flags(FlagSet):
        """
        Flags for the containing class.
        """

        flag_set_title = "Algorithm Environment"

        world_type_flag = \
            Flag(
                "--world-type",
                default = "SAT",
                choices = ("SAT", "MAX-SAT"),
                metavar = "TYPE",
                help = "world consists of TYPE tasks"
                )
        nntasks_world_flag = \
            Flag(
                "--nntasks-world",
                default = None,
                type = IntRanges,
                metavar = "NN",
                help = "use only task numbers NN [%default]",
                )

    def __init__(self, flags = Flags.given):
        """
        Initialize.
        """

        self.flags = flags
        self.supplies = NIPS2009_Supplies()

        if self.flags.world_type == "SAT":
            self.cutoffs = [
                0.5,
                2.0,
                8.0,
                32.0,
                128.0,
                512.0,
                ]
            self.outcomes = Outcomes(self, 2)
            self.samples = SAT_Samples(self)
            self.actions = WorldDescriptionActions(self, exclude_nnsolver = [9])
            self.utilities = numpy.array([1.0, 0.0])
            self.nsuccess = 0
        elif self.flags.world_type == "MAX-SAT":
            self.cutoffs = [
                1.0,
                4.0,
                16.0,
                64.0,
                256.0,
                ]
            self.samples = MAX_SAT_Samples(self)
            self.actions = WorldDescriptionActions(self)
            self.outcomes = Outcomes(self, self.nactions)
#            self.utilities = (numpy.ones(self.nactions) * 0.75)**numpy.arange(self.nactions)
            # or...
            self.utilities = numpy.zeros(self.nactions)
            self.utilities[0] = 1.0
            self.nsuccess = None
        else:
            assert False

        self.tasks = WorldDescriptionTasks(self, self.flags.nntasks_world)

    def sample_action(self, task, action):
        """
        Retrieve a random sample.
        """

        return self.samples.sample_action(task, action)

    # properties
    ntasks = property(lambda self: len(self.tasks))
    nactions = property(lambda self: len(self.actions))
    noutcomes = property(lambda self: len(self.outcomes))

