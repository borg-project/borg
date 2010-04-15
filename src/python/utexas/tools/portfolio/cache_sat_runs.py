"""
utexas/portfolio/sat_world.py

The world of SAT.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.cache_sat_runs import main

    raise SystemExit(main())

from contextlib     import closing
from sqlalchemy     import (
    Index,
    select,
    create_engine,
    )
from sqlalchemy.orm import sessionmaker
from cargo.flags    import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.log      import get_logger
from utexas.data    import (
    DatumBase,
    SAT_SolverRun,
    ResearchSession,
    research_connect,
    )

log          = get_logger(__name__)
module_flags = \
    Flags(
        "Cache Construction Options",
        Flag(
            "-v",
            "--verbose",
            action  = "store_true",
            help    = "be noisier [%default]",
            ),
        )

def build_cache(lsession, rsession):
    """
    Build a local cache of solver runs.
    """

    # query runs from remote
    log.info("storing solver runs in cache")

    statement =                                                                 \
        select(
            [
                SAT_SolverRun.uuid,
                SAT_SolverRun.task_uuid,
                SAT_SolverRun.solver_name,
                SAT_SolverRun.proc_elapsed,
                SAT_SolverRun.cutoff,
                SAT_SolverRun.satisfiable,
                ],
            )
    executed = rsession.connection().execute(statement)

    # store them to local
    lsession.connection().execute(
        SAT_SolverRun.__table__.insert(),
        [
            {
                "uuid"         : row[0],
                "task_uuid"    : row[1],
                "solver_name"  : row[2],
                "proc_elapsed" : row[3],
                "cutoff"       : row[4],
                "satisfiable"  : row[5],
                }                        \
            for row in executed
            ]
        )

    lsession.commit()

    # index them
    log.info("indexing runs in cache")

    runs_index = \
        Index(
            "sat_solver_runs_tsc_ix",
            SAT_SolverRun.task_uuid,
            SAT_SolverRun.solver_name,
            )

    runs_index.create(lsession.connection().engine)

    # done
    log.info("done building cache")

@with_flags_parsed(
    usage = "usage: %prog [options] <cache>",
    )
def main((cache_path,)):
    """
    Main.
    """

    # basic flag handling
    flags = module_flags.given

    if flags.verbose:
        get_logger("utexas.tools.sat.run_solvers").setLevel(logging.NOTSET)
        get_logger("cargo.unix.accounting").setLevel(logging.DEBUG)
        get_logger("utexas.sat.solvers").setLevel(logging.DEBUG)

    # write the cache
    ResearchSession.configure(bind = research_connect())

    local_engine         = create_engine("sqlite:///%s" % cache_path)
    LocalResearchSession = sessionmaker(bind = local_engine)

    DatumBase.metadata.create_all(local_engine)

    with closing(LocalResearchSession()) as lsession:
        with closing(ResearchSession()) as rsession:
            build_cache(lsession, rsession)

