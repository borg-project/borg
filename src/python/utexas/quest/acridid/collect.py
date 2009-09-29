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
from cargo.io import files_under
from cargo.ai.sat.solvers import ArgoSAT_Solver
from cargo.log import get_logger
from cargo.sql.alchemy import SQL_Session
from cargo.flags import (
    Flag,
    FlagSet,
    parse_given,
    with_flags_parsed,
    )
from cargo.sugar import run_once
from cargo.labor.jobs import (
    Job,
    Jobs,
    )
from cargo.labor.storage import outsource
from cargo.errors import (
    Raised,
    print_ignored_error,
    )
from utexas.quest.acridid.core import (
    SAT_Task,
    SAT_SolverRun,
    SAT_SolverDescription,
    ArgoSAT_Configuration,
    )

log = get_logger(__name__, level = None)

class ModuleFlags(FlagSet):
    """
    Flags that apply to this module.
    """

    flag_set_title = "Script Configuration"

    benchmark_root_flag = \
        Flag(
            "--benchmark-root",
            default = ".",
            metavar = "PATH",
            help    = "run on CNF instances under PATH [%default]",
            )
    outsource_flag = \
        Flag(
            "--outsource",
            action  = "store_true",
            help    = "store labor for workers",
            )

flags = ModuleFlags.given

def yield_tasks(subdirectory = ""):
    """
    Yield the task descriptions.
    """

    directory = os.path.join(flags.benchmark_root, subdirectory)

    for path in files_under(directory, "*.cnf"):
        yield (path, SAT_Task.from_file(flags.benchmark_root, path))

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
        Run this job in a transaction.
        """

        session              = SQL_Session(self.database)
        solver               = ArgoSAT_Solver(flags = ArgoSAT_Solver.Flags(argosat_path = self.solver_path))
        task                 = session.merge(self.task)
        solver_description   = session.merge(SAT_SolverDescription(name = "argosat"))
        solver_configuration = session.merge(self.configuration)
        cutoff               = timedelta(seconds = 512.0)

        run = \
            SAT_SolverRun.starting_now(
                task          = task,
                solver        = solver_description,
                configuration = solver_configuration,
                )
        (outcome, elapsed, censored) = \
            solver.solve(
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
        session.commit()

def yield_jobs():
    """
    Yield units of work.
    """

    session       = SQL_Session()
    database      = session.connection().engine.url
    configuration = ArgoSAT_Configuration.from_names("r", "r", "n")
    solver        = ArgoSAT_Solver(argv = configuration.argv)
    tasks         = [(p, session.merge(t)) for (p, t) in yield_tasks("satlib/dimacs")]

    for i in xrange(64):
        seed = numpy.random.randint(0, 2**30)

        for (path, task) in tasks:
            yield \
                RunSolverJob(
                    database      = database,
                    seed          = seed,
                    path          = path,
                    task          = task,
                    solver_path   = solver.flags.argosat_path,
                    configuration = configuration,
                    )

@with_flags_parsed()
def main(positional):
    """
    Application body.
    """

    jobs = list(yield_jobs())

    if flags.outsource:
        log.note("outsourcing %i jobs", len(jobs))

        outsource(jobs, "random argosat runs on satlib/dimacs")
    else:
        log.note("running %i jobs", len(jobs))

        Jobs(jobs).run()

