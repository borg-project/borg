"""
utexas/quest/acridid/collect.py

Collect run data.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.quest.acridid.collect import main

    raise SystemExit(main())

import os
import os.path
import sys
import logging
import numpy

from datetime import timedelta
from contextlib import closing
from cargo.io import files_under
from cargo.ai.sat.solvers import (
    ArgoSAT_Solver,
    SAT_Competition2007_Solver,
    )
from cargo.log import get_logger
from cargo.sql.alchemy import SQL_Engines
from cargo.flags import (
    Flag,
    Flags,
    parse_given,
    )
from cargo.sugar import run_once
from cargo.labor.jobs import (
    Job,
    Jobs,
    )
from cargo.labor.storage import (
    LaborSession,
    outsource,
    labor_connect,
    )
from utexas.quest.acridid.core import (
    SAT_Task,
    SAT_SolverRun,
    AcrididSession,
    SAT_SolverDescription,
    ArgoSAT_Configuration,
    SAT_2007_SolverDescription,
    acridid_connect,
    )

log          = get_logger(__name__, level = None)
script_flags = \
    Flags(
        "Script Configuration",
        Flag(
            "--benchmark-root",
            default = ".",
            metavar = "PATH",
            help    = "run on CNF instances under PATH [%default]",
            ),
        Flag(
            "--outsource",
            action  = "store_true",
            help    = "outsource labor to workers",
            ),
        )

def yield_tasks(subdirectory = ""):
    """
    Yield the task descriptions.
    """

    directory = os.path.join(script_flags.given.benchmark_root, subdirectory)

    for path in files_under(directory, "*.cnf"):
        yield (path, SAT_Task.from_file(script_flags.given.benchmark_root, path))

def root_relative(relative):
    """
    Return an absolute path given a path relative to the project root.
    """

    return os.path.normpath(os.path.join(os.environ.get("ACRIDID_ROOT", ""), relative))

class RunSolverJob(Job):
    """
    Run a solver on a task.
    """

    @staticmethod
    @run_once
    def class_set_up():
        """
        Common setup code.
        """

        # logging configuration
        get_logger("sqlalchemy.engine").setLevel(logging.WARNING)
        get_logger("cargo.ai.sat.solvers").setLevel(logging.DEBUG)

    def run(self):
        """
        Run this job.
        """

        # database configuration
        session = AcrididSession(bind = SQL_Engines.default.get(self.database))

        with closing(session):
            self.run_in_transaction(session)
            session.commit()

    def run_in_transaction(self, session):
        """
        Run this job in a transaction.
        """

        cutoff  = timedelta(seconds = 512.0)

        if self.configuration is None:
            configuration = None
        else:
            configuration = session.merge(self.configuration)

        run = \
            SAT_SolverRun.starting_now(
                task          = session.merge(self.task),
                solver        = session.merge(SAT_SolverDescription(name = "argosat")),
                configuration = configuration,
                )
        (outcome, elapsed, censored) = \
            self.solver.solve(
                cutoff,
                self.path,
                self.seed,
                )

        log.info("solver returned %s after %s", outcome, elapsed)

        run.outcome  = outcome
        run.elapsed  = elapsed
        run.cutoff   = cutoff
        run.censored = censored
        run.seed     = self.seed

        session.add(run)

def yield_argosat_seed_jobs():
    """
    Yield units of work.
    """

    session       = SQL_Session()
    configuration = ArgoSAT_Configuration.from_names("r", "r", "n")
    solver        = ArgoSAT_Solver(argv = configuration.argv)
    tasks         = [(p, session.merge(t)) for (p, t) in yield_tasks("satlib/dimacs")]

    for i in xrange(64):
        seed = numpy.random.randint(0, 2**30)

        for (path, task) in tasks:
            yield \
                RunSolverJob(
                    database      = session.connection().engine.url,
                    seed          = seed,
                    path          = path,
                    task          = task,
                    solver        = solver,
                    configuration = configuration,
                    )

def yield_sat2007_random_jobs():
    """
    Yield units of work.
    """

    session = AcrididSession()

    with closing(session):
        tasks = [(p, session.merge(t)) for (p, t) in yield_tasks("sat2007/random")]

        for solver_description in session.query(SAT_2007_SolverDescription).all():
            solver = \
                SAT_Competition2007_Solver(
                    solver_description.relative_path,
                    solver_description.seeded,
                    )

            for (path, task) in tasks:
                yield \
                    RunSolverJob(
                        database      = session.connection().engine.url,
                        seed          = None,
                        path          = path,
                        task          = task,
                        solver        = solver,
                        configuration = None,
                        )

        session.commit()

def main():
    """
    Application body.
    """

    parse_given()

    AcrididSession.configure(bind = acridid_connect())

    get_logger("cargo.labor.storage").setLevel(logging.NOTE)

    with SQL_Engines.default:
        jobs = list(yield_sat2007_random_jobs())

        if script_flags.given.outsource:
            LaborSession.configure(bind = labor_connect())

            outsource(jobs, "run SAT-2007 competition solvers on the random benchmark")
        else:
            log.note("running %i jobs", len(jobs))

            Jobs(jobs).run()

