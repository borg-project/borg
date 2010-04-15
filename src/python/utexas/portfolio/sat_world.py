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
from sqlalchemy.sql.functions import (
    count,
    random as sql_random,
    )
from cargo.log                import get_logger
from cargo.flags              import (
    Flag,
    Flags,
    )
from cargo.iterators          import grab
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

    def __init__(self, n, task):
        """
        Initialize.

        @param task: SAT task description.
        """

        self.n = n
        self.task = task

    def __str__(self):
        """
        Return a human-readable description of this task.
        """

        return "%s" % (self.task.uuid,)

class SAT_Outcome(Outcome):
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

    @staticmethod
    def from_run(run):
        """
        Return an outcome from a solver run.
        """

        return SAT_Outcome.from_bool(run.outcome)

    @staticmethod
    def from_bool(bool):
        """
        Return an outcome from True, False, or None.
        """

        return SAT_Outcome.BY_VALUE[bool]

# outcome constants
SAT_Outcome.SOLVED   = SAT_Outcome(0, 1.0)
SAT_Outcome.UNKNOWN  = SAT_Outcome(1, 0.0)
SAT_Outcome.BY_VALUE = {
    True:  SAT_Outcome.SOLVED,
    False: SAT_Outcome.SOLVED,
    None:  SAT_Outcome.UNKNOWN,
    }
SAT_Outcome.BY_INDEX = (
    SAT_Outcome.SOLVED,
    SAT_Outcome.UNKNOWN,
    )

class SAT_World(World):
    """
    Components of the SAT world.
    """

    def __init__(self, actions, tasks, flags = module_flags.given):
        """
        Initialize.
        """

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

