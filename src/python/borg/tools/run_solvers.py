"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                   import call
    from borg.tools.run_solvers import main

    call(main)

from uuid           import UUID
from plac           import annotations
from cargo.log      import get_logger
from cargo.sugar    import composed
from cargo.temporal import parse_timedelta

log = get_logger(__name__, default_level = "INFO")

class SolveTaskJob(object):
    """
    Attempt to solve a given task.
    """

    def __init__(self, url, solver, task_uuid, budget, named, recycle):
        """
        Initialize.
        """

        from cargo.random import get_random_random
        from borg.tasks   import get_collections

        self._url         = url
        self._solver      = solver
        self._task_uuid   = task_uuid
        self._budget      = budget
        self._named       = named
        self._recycle     = recycle
        self._random      = get_random_random()
        self._collections = get_collections()

    def __call__(self):
        """
        Make some number of solver runs on a task.
        """

        # log as appropriate
        get_logger("cargo.unix.accounting",      level = "DEBUG")
        get_logger("borg.solvers.competition",   level = "DEBUG")
        get_logger("borg.solvers.uncompressing", level = "DEBUG")

        # connect to the database
        from cargo.sql.alchemy import SQL_Engines

        MainSession = SQL_Engines.default.make_session(self._url)

        with MainSession() as session:
            # set up the environment
            from borg.solvers import Environment

            environment = \
                Environment(
                    MainSession   = MainSession,
                    named_solvers = self._named,
                    collections   = self._collections,
                    )

            # prepare the run
            from borg.data import TaskRow

            task_row = session.query(TaskRow).get(self._task_uuid)

            if self._recycle:
                from borg.tasks import UUID_Task

                full_solver = self._solver
                task        = UUID_Task(self._task_uuid, row = task_row)
            else:
                from borg.solvers import UncompressingSolver

                full_solver = UncompressingSolver(self._solver)
                task        = task_row.get_task(environment)

            # make the run
            log.info("running %s on %s", self._solver.name, task_row.uuid)

            session.commit()

            attempt = full_solver.solve(task, self._budget, self._random, environment)

            # store the attempt
            attempt.get_row(session)

            session.commit()

    @staticmethod
    @composed(list)
    def make_all(session, budget, restarts, seeded_restarts, recycle, names = None):
        """
        Generate a set of jobs to distribute.
        """

        from borg.data    import (
            TaskRow as TR,
            SolverRow as SR,
            )
        from borg.solvers import (
            Environment,
            LookupSolver,
            get_named_solvers,
            )

        named_solvers = get_named_solvers(use_recycled = recycle)
        environment   = Environment(named_solvers = named_solvers)
        task_uuids    = [u for (u,) in session.query(TR.uuid)]

        if names is None:
            names = [n for (n,) in session.query(SR.name)]

        for name in names:
            solver = LookupSolver(name)

            if solver.get_seeded(environment):
                restarts_of = max(restarts, seeded_restarts)
            else:
                restarts_of = restarts

            log.info("making %i restarts of %s", restarts_of, solver.name)

            for task_uuid in task_uuids:
                for i in xrange(restarts_of):
                    yield SolveTaskJob(
                        session.connection().engine.url,
                        solver,
                        task_uuid,
                        budget,
                        named_solvers,
                        recycle,
                        )

@annotations(
    budget          = ("solver budget"    , "positional", None , parse_timedelta),
    restarts        = ("minimum attempts" , "option"    , "r"  , int)            ,
    seeded_restarts = ("minimum attempts" , "option"    , "s"  , int)            ,
    run_only        = ("run one solver"   , "option")   ,
    recycle         = ("reuse past runs"  , "flag")     ,
    outsource       = ("outsource labor"  , "flag")     ,
    )
def main(
    budget,
    restarts        = 1,
    seeded_restarts = 1,
    run_only        = None,
    recycle         = False,
    outsource       = False,
    ):
    """
    Run the script.
    """

    # enable log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    # connect to the database and go
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        from cargo.sql.alchemy import make_session
        from borg.data         import research_connect

        ResearchSession = make_session(bind = research_connect())

        with ResearchSession() as session:
            jobs = \
                SolveTaskJob.make_all(
                    session,
                    budget,
                    restarts,
                    seeded_restarts,
                    recycle,
                    names = None if run_only is None else [run_only],
                    )

        # run the jobs
        from cargo.labor import outsource_or_run

        outsource_or_run(jobs, outsource)

