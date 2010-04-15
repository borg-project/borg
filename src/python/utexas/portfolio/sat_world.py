"""
utexas/portfolio/sat_world.py

The world of SAT.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from itertools                import izip
from contextlib               import closing
from collections              import (
    Sequence,
    defaultdict,
    )
from sqlalchemy               import (
    and_,
    case,
    select,
    create_engine,
    )
from sqlalchemy.orm           import sessionmaker
from sqlalchemy.sql.functions import random as sql_random
from cargo.log                import get_logger
from cargo.flags              import (
    Flag,
    Flags,
    )
from cargo.iterators          import grab
from utexas.sat.solvers       import get_random_seed
from utexas.data              import (
    DatumBase,
    SAT_SolverRun,
    ResearchSession,
    )
from utexas.portfolio.world   import (
    Task,
    World,
    Action,
    Outcome,
    )

log          = get_logger(__name__)
module_flags = \
    Flags(
        "SAT Data Storage",
        Flag(
            "--sat-world-cache",
            default = "sqlite:///:memory:",
            metavar = "DATABASE",
            help    = "use research DATABASE by default [%default]",
            ),
        )

class SAT_WorldAction(Action):
    """
    An action in the world.
    """

    def __init__(self, n, solver, cutoff):
        """
        Initialize.
        """

        self.n = n
        self.solver        = solver
        self.cutoff        = cutoff

    def __str__(self):
        """
        Return a human-readable description of this action.
        """

        return "%s_%ims" % (self.solver.name, int(self.cutoff.as_s * 1000))

class SAT_WorldTask(Task):
    """
    A task in the world.
    """

    def __init__(self, path, name = None):
        """
        Initialize.

        @param task: SAT task description.
        """

        self.path = path
        self.name = name

    def __str__(self):
        """
        Return a human-readable description of this task.
        """

        if self.name is None:
            return self.path
        else:
            return self.name

    def __json__(self):
        """
        Make JSONable.
        """

        return (self.path, self.name)

class SAT_WorldAction(Action):
    """
    An action in the world.
    """

    def __init__(self, solver, cutoff):
        """
        Initialize.
        """

        self.solver   = solver
        self.cutoff   = cutoff
        self.cost     = cutoff
        self.outcomes = SAT_WorldOutcome.BY_INDEX

    def __str__(self):
        """
        Return a human-readable description of this action.
        """

        return "%s_%ims" % (self.solver.name, int(self.cutoff.as_s * 1000))

    def __json__(self):
        """
        Make JSONable.
        """

        return (self.solver.name, self.cutoff.as_s)

    def take(self, task, random = numpy.random):
        """
        Take the action.
        """

        if self.solver.seeded:
            seed = get_random_seed(random)
        else:
            seed = None

        result  = self.solver.solve(task.path, self.cutoff, seed = None)
        outcome = SAT_WorldOutcome.from_result(result)

        return (outcome, result)

class SAT_WorldOutcome(Outcome):
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

    def __json__(self):
        """
        Make JSONable.
        """

        return self.n

    @staticmethod
    def from_result(result):
        """
        Return an outcome from a solver result.
        """

        return SAT_WorldOutcome.from_bool(result.satisfiable)

    @staticmethod
    def from_bool(bool):
        """
        Return an outcome from True, False, or None.
        """

        return SAT_WorldOutcome.BY_VALUE[bool]

# outcome constants
SAT_WorldOutcome.SOLVED   = SAT_WorldOutcome(0, 1.0)
SAT_WorldOutcome.UNSOLVED = SAT_WorldOutcome(1, 0.0)
SAT_WorldOutcome.BY_VALUE = {
    True:  SAT_WorldOutcome.SOLVED,
    False: SAT_WorldOutcome.SOLVED,
    None:  SAT_WorldOutcome.UNSOLVED,
    }
SAT_WorldOutcome.BY_INDEX = [
    SAT_WorldOutcome.SOLVED,
    SAT_WorldOutcome.UNSOLVED,
    ]


class SAT_World(object):
    # FIXME
    def __init__(self, actions, tasks, flags = module_flags.given):
        these_flags = module_flags.merged(flags)

        # world properties
        self.actions   = actions
        self.tasks     = tasks
        self.outcomes  = SAT_Outcome.BY_INDEX
        self.utilities = numpy.array([o.utility for o in self.outcomes])

        # establish a local sqlite cache
        # FIXME engine disposal?
        self.LocalResearchSession = sessionmaker(bind = create_engine(these_flags.sat_world_cache))

    def act(self, task, action, nrestarts = 1, random = numpy.random):
        """
        Retrieve a random sample.
        """

        log.debug("taking %i action(s) %s on task %s", nrestarts, action, task)

        return [o for (o, _) in self.act_extra(task, action, nrestarts, random)]

    def act_extra(self, task, action, nrestarts = 1, random = numpy.random):
        """
        Act, and provide a "true" value for time spent.
        """

        sat_case  = [(SAT_SolverRun.proc_elapsed <= action.cutoff, SAT_SolverRun.satisfiable)]
        statement = \
            select(
                [
                    case(sat_case),
                    SAT_SolverRun.proc_elapsed,
                    ],
                and_(
                    SAT_SolverRun.task_uuid   == task.task.uuid,
                    SAT_SolverRun.solver_name == action.solver.name,
                    SAT_SolverRun.cutoff      >= action.cutoff,
                    ),
                )

        with closing(self.LocalResearchSession()) as lsession:
            rows  = lsession.connection().execute(statement)
            pairs = [(SAT_Outcome.BY_VALUE[s], min(e, action.cutoff)) for (s, e) in rows]

            return [grab(pairs, random) for i in xrange(nrestarts)]

    def act_once_extra(self, task, action):
        """
        Act, and provide a "true" value for time spent.
        """

        sat_case  = [(SAT_SolverRun.proc_elapsed <= action.cutoff, SAT_SolverRun.satisfiable)]
        statement = \
            select(
                [
                    case(sat_case),
                    SAT_SolverRun.proc_elapsed,
                    ],
                and_(
                    SAT_SolverRun.task_uuid   == task.task.uuid,
                    SAT_SolverRun.solver_name == action.solver.name,
                    SAT_SolverRun.cutoff      >= action.cutoff,
                    ),
                order_by = sql_random(),
                limit    = 1,
                )

        with closing(self.LocalResearchSession()) as lsession:
            ((sat, elapsed),) = lsession.connection().execute(statement)

            return (SAT_Outcome.BY_VALUE[sat], min(elapsed, action.cutoff))

    def get_true_probabilities(self, task, action):
        """
        Get the true outcome probabilities for an action.
        """

        sat_case  = [(SAT_SolverRun.proc_elapsed <= action.cutoff, SAT_SolverRun.satisfiable)]
        statement = \
            select(
                [case(sat_case)],
                and_(
                    SAT_SolverRun.task_uuid   == task.task.uuid,
                    SAT_SolverRun.solver_name == action.solver.name,
                    SAT_SolverRun.cutoff      >= action.cutoff,
                    ),
                )

        with closing(self.LocalResearchSession()) as lsession:
            rows   = lsession.connection().execute(statement)
            counts = numpy.zeros(len(self.outcomes))

            for (sat,) in rows:
                counts[SAT_Outcome.BY_VALUE[sat].n] += 1

        return counts / numpy.sum(counts)

