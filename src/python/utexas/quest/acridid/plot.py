"""
utexas/quest/acridid/argo_plot.py

General support routines.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import logging

from uuid import UUID
from datetime import timedelta
from itertools import (
    izip,
    product,
    )
from sqlalchemy import (
    and_,
    Column,
    Binary,
    String,
    Integer,
    Boolean,
    Interval,
    ForeignKey,
    )
from sqlalchemy.orm import relation
from sqlalchemy.orm.exc import NoResultFound
from cargo.io import (
    hash_file,
    files_under,
    )
from cargo.ai.sat.solvers import ArgoSAT_Solver
from cargo.log import get_logger
from cargo.sql.alchemy import (
    SQL_Base,
    SQL_UUID,
    SQL_Session,
    UTC_DateTime,
    )
from cargo.flags import (
    Flag,
    FlagSet,
    with_flags_parsed,
    )
from cargo.errors import print_ignored_error
from cargo.temporal import utc_now
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

def get_argosat_solver_description(session):
    """
    Get the ArgoSAT description.
    """

    argosat_description = SAT_SolverDescription(name = "argosat")

    return session.merge(argosat_description)

def yield_argosat_configurations(session):
    """
    Yield the relevant possible ArgoSAT configurations.
    """

    vs_strategies = {
        "r":    ("random",),
#         "m":    ("minisat", "init", "1.0", "1.052"),
#         "r+m":  ("random", "0.05", "minisat", "init", "1.0", "1.052"),
        }
    ps_strategies = {
#         "t": ("true",),
#         "f": ("false",),
        "r": ("random", "0.5"),
        }
    rs_strategies = {
        "n": ("no_restart",),
        }

    uuids = [
        UUID("72f8c280c63f4d81a00473850a34b710"),
#         UUID("ac29aaaf043845d5a49bdf2d2682bd54"),
#         UUID("7a8288b195314aa980e1e7276e7da4cc"),
#         UUID("e442e7c8b9b94fbe8603663645f741ff"),
#         UUID("dc92495317104e3fbb99ad5c582872ce"),
#         UUID("d9b44148f74b4a6ea61bd222aaeed1cf"),
#         UUID("6eb47bfab181429e9c255620130bc529"),
#         UUID("523d6459091c4b91b76bfbaff8c12096"),
#         UUID("1f53675770e2488b99bc83ae5789a17e"),
        ]

    s = \
        izip(
            uuids,
            product(
                vs_strategies.iteritems(),
                ps_strategies.iteritems(),
                rs_strategies.iteritems(),
                )
            )

    for (uuid, ((vss_name, vss), (pss_name, pss), (rss_name, rss))) in s:
        name    = "v:%s,p:%s,r:%s" % (vss_name, pss_name, rss_name)
        c       = \
            ArgoSAT_Configuration(
                uuid               = uuid,
                name               = name,
                variable_selection = vss,
                polarity_selection = pss,
                restart_scheduling = rss,
                )

        yield session.merge(c)

def yield_tasks(session):
    """
    Yield the task descriptions.
    """

    for path in files_under(flags.benchmark_root, "*.cnf"):
        # hash and retrieve the task
        relative_path   = os.path.relpath(path, flags.benchmark_root)
        (_, hash)       = hash_file(path, "sha512")
        query           = \
            session.query(SAT_Task).filter(
                and_(
                    SAT_Task.hash == hash,
                    SAT_Task.path == relative_path,
                    )
                )

        try:
            task = query.one()
        except NoResultFound:
            task = SAT_Task(path = relative_path, name = os.path.basename(path), hash = hash)

            log.info("adding %s", relative_path)
            session.add(task)

        yield task

def run_solver_on(session, solver_description, configuration, task):
    solver = ArgoSAT_Solver(argv = configuration.argv)
    run    = \
        SAT_SolverRun.starting_now(
            task          = task,
            solver        = solver_description,
            configuration = configuration,
            )
    path                               = os.path.join(flags.benchmark_root, task.path)
    (outcome, elapsed, censored, seed) = solver.solve(timedelta(seconds = 32.0), path)

    log.info("solver returned %s after %s", outcome, elapsed)

    run.outcome  = outcome
    run.elapsed  = elapsed
    run.censored = censored
    run.seed     = seed

    session.add(run)

@with_flags_parsed()
def main(positional):
    """
    Application body.
    """

    # configure logging
    get_logger("sqlalchemy.engine").setLevel(logging.INFO)
    get_logger("cargo.ai.sat.solvers").setLevel(logging.DEBUG)

    # run the experiment
    session = SQL_Session()

    try:
        tasks               = list(yield_tasks(session))
        configurations      = list(yield_argosat_configurations(session))
        argosat_description = get_argosat_solver_description(session)

        session.commit()

        for task in (t for t in tasks if t.path.startswith("parity/")):
            run_solver_on(session, argosat_description, configurations[0], task)

            session.commit()
    finally:
        try:
            session.rollback()
        except:
            print_ignored_error()

import math
import pprint
import pylab
from collections import defaultdict

@with_flags_parsed()
def plot_main(positional):
    """
    Application body.
    """

    session             = SQL_Session()
    configuration       = list(yield_argosat_configurations(session))[0]
    argosat_description = get_argosat_solver_description(session)

    query = \
        session.query(SAT_SolverRun).filter(
            and_(
                SAT_SolverRun.solver == argosat_description,
                SAT_SolverRun.configuration == configuration,
                )
            )

    times = defaultdict(list)

    for run in query:
        if run.task.path.startswith("parity/"):
            log_time = math.log(1.0 + run.elapsed.seconds + run.elapsed.microseconds / 1e6)

            times[run.task.path].append(log_time)

    pprint.pprint(dict(times))

if __name__ == '__main__':
#     main()
    plot_main()

