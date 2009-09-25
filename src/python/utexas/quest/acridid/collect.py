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
    SQL_Job,
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

class RunSolverJob(SQL_Job):
    """
    Run a solver on a task.
    """

    def __init__(self):
        """
        Initialize.
        """

        pass

    @staticmethod
    @run_once
    def class_set_up():
        """
        Common setup code.
        """

        # logging configuration
        get_logger("sqlalchemy.engine").setLevel(logging.WARNING)
        get_logger("cargo.ai.sat.solvers").setLevel(logging.DEBUG)

    @staticmethod
    @run_once
    def class_tear_down():
        """
        Common teardown code.
        """

        pass

    def run(self):
        """
        Run this job in a transaction.
        """

        session              = self.session
        task                 = session.merge(self.task)
        solver_description   = session.merge(SAT_SolverDescription(name = "argosat"))
        solver_configuration = session.merge()
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

def yield_jobs(session):
    """
    Yield units of work.
    """

    configuration = ArgoSAT_Configuration.from_names("r", "r", "n")
    solver        = ArgoSAT_Solver(argv = solver_configuration.argv)
    tasks         = [(p, session.merge(t)) for (p, t) in yield_tasks("satlib/dimacs")]

    for i in xrange(64):
        seed = numpy.random.randint(0, 2**30)

        for (path, task) in tasks:
            yield \
                RunSolverJob(
                    seed          = seed,
                    path          = path,
                    task          = task,
                    solver        = solver,
                    configuration = configuration,
                    )

import time

class SleepJob(Job):
    def run(self):
        log.warning("sleeping!")
        time.sleep(8)

@with_flags_parsed()
def main(positional):
    """
    Application body.
    """

#     argv    = [
#         "--database",
#         "postgresql://postgres@zerogravitas.csres.utexas.edu:5432/acridid-20090921",
#         ]
#     matching = "InMastodon && ( Arch == \"INTEL\" ) && ( OpSys == \"LINUX\" ) && regexp(\"rhavan-.*\", ParallelSchedulingGroup)"

    # distribute jobs
    session = SQL_Session()
#     jobs    = list(yield_jobs(session))
    jobs = [SleepJob(), SleepJob(), SleepJob()]

#     if flags.outsource:
    outsource(jobs)
#     else:
#         Jobs(jobs).run()

