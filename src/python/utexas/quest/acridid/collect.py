"""
utexas/quest/acridid/collect.py

Collect run data.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

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
from cargo.errors import print_ignored_error
from cargo.condor.spawn import (
    CondorJob,
    CondorJobs,
    CondorSubmission,
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

class RunSolverJob(CondorJob):
    """
    Run a solver on a task.
    """

    def __init__(self, seed, path, task):
        """
        Initialize.
        """

        self.seed = seed
        self.path = path
        self.task = task

    def run(self, session):
        """
        Run this job.
        """

        task                 = session.merge(self.task)
        solver_description   = session.merge(SAT_SolverDescription(name = "argosat"))
        solver_configuration = session.merge(ArgoSAT_Configuration.from_names("r", "r", "n"))
        solver               = ArgoSAT_Solver(argv = solver_configuration.argv)
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

class CollectionJob(CondorJobs):
    """
    Collect data.
    """

    def run(self):
        """
        Initialize.
        """

        # parse arguments
#         parse_given()

        # configure logging
        get_logger("sqlalchemy.engine").setLevel(logging.WARNING)
        get_logger("cargo.ai.sat.solvers").setLevel(logging.DEBUG)

        # run the experiment
        session = SQL_Session()

        try:
            for job in self.jobs:
                job.run(session)

                session.commit()
        except:
            try:
                log.error("unhandled exception; rolling back latest transaction")

                session.rollback()
            except:
                print_ignored_error()

def yield_jobs():
    """
    Yield (possibly) parallel jobs in this script.
    """

    session = SQL_Session()
    tasks   = [(p, session.merge(t)) for (p, t) in yield_tasks("satlib/dimacs")]

    for i in xrange(64):
        seed  = numpy.random.randint(0, 2**30)

        for (path, task) in tasks:
            yield RunSolverJob(seed = seed, path = path, task = task)

@with_flags_parsed()
def main(positional):
    """
    Application body.
    """

    argv    = [
        "--benchmark-root",
        root_relative("../../tasks"),
        "--argosat-path",
        root_relative("dep/third/argosat/src/argosat"),
        "--database",
        "postgresql://postgres@zerogravitas.csres.utexas.edu:5432/acridid-20090921",
        ]
    matching = "InMastodon && ( Arch == \"INTEL\" ) && ( OpSys == \"LINUX\" ) && regexp(\"rhavan-.*\", ParallelSchedulingGroup)"

    # distribute jobs
    njobs = 128
    jobs  = [CollectionJob() for i in xrange(njobs)]

    for (i, job) in enumerate(yield_jobs()):
        jobs[i % njobs].jobs.append(job)

    for job in jobs:
        log.info("distributing %i jobs to process %s", len(job.jobs), job.uuid)

    # condor!
    submission = \
        CondorSubmission(
            jobs        = jobs,
            matching    = matching,
            argv        = argv,
            description = "sampling randomized heuristic solver outcome distributions",
            )

    submission.run_or_submit()

