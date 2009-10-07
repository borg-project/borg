"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.quest.acridid.scripts.z091007_1255_get_satenstein_runs import main

    raise SystemExit(main())

import os
import os.path
import sys
import logging
import numpy

from uuid import UUID
from datetime import timedelta
from contextlib import closing
from collections import namedtuple
from cargo.io import files_under
from cargo.ai.sat.solvers import (
    SATensteinSolver,
    )
from cargo.log import get_logger
from cargo.sql.alchemy import SQL_Engines
from cargo.flags import parse_given
from cargo.sugar import run_once
from cargo.labor.jobs import Job
from cargo.labor.storage import (
    outsource_or_run,
    )
from utexas.quest.acridid.core import (
    AcrididSession,
    SAT_ConfigurationSet,
    SAT_SolverDescription,
    SATensteinConfiguration,
    acridid_connect,
    )

log = get_logger(__name__, level = None)

class RunJob(Job):
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

        cutoff                       = timedelta(seconds = 512.0)
        seed                         = numpy.random.randint(0, 2**30)
        run                          = \
            SAT_SolverRun.starting_now(
                task          = session.merge(self.task),
                solver        = session.merge(self.description),
                configuration = session.merge(self.configuration),
                )
        (outcome, elapsed, censored) = \
            self.solver.solve(
                cutoff = cutoff,
                path   = self.path,
                seed   = seed
                )

        log.info("solver returned %s after %s", outcome, elapsed)

        run.outcome  = outcome
        run.elapsed  = elapsed
        run.cutoff   = cutoff
        run.censored = censored
        run.seed     = seed

        session.add(run)

def yield_tasks(subdirectory = ""):
    """
    Yield the task descriptions.
    """

    root      = "."
    directory = os.path.join(root, subdirectory)

    for path in files_under(directory, "*.cnf"):
        yield (path, SAT_Task.from_file(root, path))

def yield_jobs():
    """
    Yield units of work.
    """

    session = AcrididSession()

    with closing(session):
        configuration_set   =                                                                  \
            session                                                                            \
            .query(SAT_ConfigurationSet)                                                       \
            .filter(SAT_ConfigurationSet.uuid == UUID("b5bad358-baa0-4841-a735-650016591b19")) \
            .one()
        solver_description  =                                   \
            session                                             \
            .query(SAT_SolverDescription)                       \
            .filter(SAT_SolverDescription.name == "satenstein") \
            .one()
        configuration_query =                                         \
            session                                                   \
            .query(SATensteinConfiguration)                           \
            .filter(SATensteinConfiguration.set == configuration_set)
        tasks               = [(p, session.merge(t)) for (p, t) in yield_tasks("sat2007/random")]

        for configuration in configuration_query:
            solver = SATensteinSolver(parameters = configuration.parameters)

            for (path, task) in tasks:
                yield \
                    RunSolverJob(
                        database      = session.connection().engine.url,
                        path          = path,
                        task          = task,
                        solver        = solver,
                        configuration = configuration,
                        description   = solver_description,
                        )

        session.commit()

def main():
    """
    Application body.
    """

    parse_given()

    with SQL_Engines.default:
        get_logger("cargo.labor.storage").setLevel(logging.NOTE)

        AcrididSession.configure(bind = acridid_connect())

        outsource_or_run(yield_jobs())

