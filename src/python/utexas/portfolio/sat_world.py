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
    )
from sqlalchemy.sql.functions import (
    count,
    random as sql_random,
    )
from cargo.log                import get_logger
from cargo.flags              import (
    Flag,
    FlagSet,
    IntRanges,
    )
from utexas.data              import (
    SAT_SolverRun,
    ResearchSession,
    )
from utexas.portfolio.world   import (
    Task,
    World,
    Action,
    Outcome,
    )

log = get_logger(__name__)

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

        return "%s" % (self.task.path,)

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

    def __init__(self, actions, tasks):
        """
        Initialize.
        """

        self.actions   = actions
        self.tasks     = tasks
        self.outcomes  = SAT_Outcome.BY_INDEX
        self.utilities = numpy.array([o.utility for o in self.outcomes])
        self.matrix    = self.__get_outcome_matrix()

    def __get_outcome_matrix(self):
        """
        Build a matrix of outcome probabilities.
        """

        log.info("building task-action-outcome matrix")

        # hit the database
        with closing(ResearchSession()) as session:
            counts  = numpy.zeros((self.ntasks, self.nactions, self.noutcomes))

            for action in self.actions:
                run_case  = case([(SAT_SolverRun.proc_elapsed <= action.cutoff, SAT_SolverRun.satisfiable)])
                statement =                                                           \
                    select(
                        [
                            SAT_SolverRun.task_uuid,
                            run_case.label("result"),
                            count(),
                            ],
                        and_(
                            SAT_SolverRun.task_uuid.in_([t.task.uuid for t in self.tasks]),
                            SAT_SolverRun.solver        == action.solver,
                            SAT_SolverRun.cutoff        >= action.cutoff,
                            ),
                        )                                                             \
                        .group_by(SAT_SolverRun.task_uuid, "result")
                executed  = session.connection().execute(statement)

                # build the matrix
                world_tasks = dict((t.task.uuid, t) for t in self.tasks)
                total_count = 0

                for (task_uuid, result, nrows) in executed:
                    # map storage instances to world instances
                    world_task    = world_tasks[task_uuid]
                    world_outcome = SAT_Outcome.BY_VALUE[result]

                    # record the outcome count
                    counts[world_task.n, action.n, world_outcome.n] += nrows
                    total_count                                     += nrows

                if total_count == 0:
                    log.warning("no rows found for action %s", action, t.task.uuid)

            norms = numpy.sum(counts, 2, dtype = numpy.float)

            return counts / norms[:, :, numpy.newaxis]

    def act(self, task, action, nrestarts = 1, random = numpy.random):
        """
        Retrieve a random sample.
        """

        log.debug("taking %i action(s) %s on task %s", nrestarts, action, task)

        nnoutcome = random.multinomial(nrestarts, self.matrix[task.n, action.n, :])
        outcomes  = sum(([self.outcomes[i]] * n for (i, n) in enumerate(nnoutcome)), [])

        return outcomes

    def act_once_extra(self, task, action):
        """
        Act, and provide a "true" value for time spent.
        """

        # FIXME we want to provide the true action
        return (self.act_once(task, action), action.cutoff)

