"""
utexas/quest/acridid/collect.py

Collect run data.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

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
    with_flags_parsed,
    )
from cargo.errors import print_ignored_error
from cargo.condor.spawn import (
    CondorJob,
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

def run_job(seed = None):
    """
    Run a job, typically under condor.
    """

    # configure logging
    get_logger("sqlalchemy.engine").setLevel(logging.WARNING)
    get_logger("cargo.ai.sat.solvers").setLevel(logging.DEBUG)

    # run the experiment
    session = SQL_Session()

    try:
        tasks                = [(p, session.merge(t)) for (p, t) in yield_tasks("dimacs/parity")]
        solver_description   = session.merge(SAT_SolverDescription(name = "argosat"))
        solver_configuration = session.merge(ArgoSAT_Configuration.from_names("r", "r", "n"))
        solver               = ArgoSAT_Solver(argv = solver_configuration.argv)

        for (path, task) in tasks:
            run = \
                SAT_SolverRun.starting_now(
                    task          = task,
                    solver        = solver_description,
                    configuration = solver_configuration,
                    )
            (outcome, elapsed, censored) = \
                solver.solve(
                    timedelta(seconds = 32.0),
                    path,
                    seed,
                    )

            log.info("solver returned %s after %s", outcome, elapsed)

            run.outcome  = outcome
            run.elapsed  = elapsed
            run.censored = censored
            run.seed     = seed

            session.add(run)
            session.commit()
    except:
        try:
            log.error("unhandled exception; rolling back transaction")

            session.rollback()
        except:
            print_ignored_error()

def yield_jobs():
    """
    Yield (possibly) parallel jobs in this script.
    """

    for i in xrange(16):
        seed = numpy.random.randint(0, 2**30)

        log.info("the random seed for job %i is %i", i, seed)

        yield CondorJob(run_job, seed = seed)

@with_flags_parsed()
def main(positional):
    """
    Application body.
    """

    matching   = "InMastodon && ( Arch == \"INTEL\" ) && ( OpSys == \"LINUX\" ) && regexp(\"rhavan-.*\", ParallelSchedulingGroup)"
    submission = \
        CondorSubmission(
            jobs        = list(yield_jobs()),
            matching    = matching,
            description = "sampling randomized heuristic solver outcome distributions",
            )

    submission.run_or_submit()

if __name__ == '__main__':
    __name__ = "utexas.quest.acridid.collect"

    main()

