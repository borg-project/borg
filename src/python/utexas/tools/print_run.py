"""
utexas/tools/print_run.py

Print a solver run.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.print_run import main

    raise SystemExit(main())

from cargo.log   import get_logger
from cargo.flags import (
    Flag,
    Flags,
    parse_given,
    )

log          = get_logger(__name__, "NOTSET")
module_flags = \
    Flags(
        "Run Printing",
        Flag(
            "--run-uuid",
            metavar = "UUID",
            help    = "print run UUID [%default]",
            ),
        )

def print_preprocessor_run(session, run_uuid):
    """
    Print a specific preprocessor run.
    """

    from utexas.data import SAT_PreprocessorRunRecord

    runs =                                                  \
        session                                             \
        .query(SAT_PreprocessorRunRecord)                   \
        .filter(SAT_PreprocessorRunRecord.uuid == run_uuid)

    for run in runs:
        log.info("PREPROCESSOR RUN: %s", run.uuid)
        log.info("preprocessor_name: %s", run.preprocessor_name)
        log.info("started: %s", run.started)
        log.info("usage_elapsed: %s", run.usage_elapsed)
        log.info("proc_elapsed: %s", run.proc_elapsed)
        log.info("cutoff: %s", run.cutoff)
        log.info("fqdn: %s", run.fqdn)
        log.info(
            "stdout follows (%i characters)\n%s",
            len(run.stdout),
            run.stdout,
            )
        log.info(
            "stderr follows (%i characters)\n%s",
            len(run.stderr),
            run.stderr,
            )
        log.info("exit_status: %s", run.exit_status)
        log.info("exit_signal: %s", run.exit_signal)

def print_solver_run(session, run_uuid):
    """
    Print a specific solver run.
    """

    from utexas.data import (
        SAT_SolverRunRecord,
        TaskNameRecord,
        )

    runs =                                            \
        session                                       \
        .query(SAT_SolverRunRecord)                   \
        .filter(SAT_SolverRunRecord.uuid == run_uuid)

    for run in runs:
        if run.preprocessor_run_uuid is not None:
            print_preprocessor_run(session, run.preprocessor_run_uuid)

        log.info("SOLVER RUN: %s", run.uuid)
        log.info("task_uuid: %s", run.task_uuid)

        descriptions =                               \
            session                                  \
            .query(TaskNameRecord)                   \
            .filter(TaskNameRecord.task == run.task)

        for description in descriptions:
            log.info(
                "in collection %s task is %s",
                description.collection,
                description.name,
                )

        log.info("solver_name: %s", run.solver_name)
        log.info("started: %s", run.started)
        log.info("usage_elapsed: %s", run.usage_elapsed)
        log.info("proc_elapsed: %s", run.proc_elapsed)
        log.info("cutoff: %s", run.cutoff)
        log.info("fqdn: %s", run.fqdn)
        log.info("seed: %s", run.seed)
        log.info(
            "stdout follows (%i characters)\n%s",
            len(run.stdout),
            run.stdout,
            )
        log.info(
            "stderr follows (%i characters)\n%s",
            len(run.stderr),
            run.stderr,
            )
        log.info("exit_status: %s", run.exit_status)
        log.info("exit_signal: %s", run.exit_signal)
        log.info("satisfiable: %s", run.satisfiable)

def main():
    """
    Application body.
    """

    # get command line arguments
    import utexas.data

    from cargo.sql.alchemy import SQL_Engines
    from cargo.flags       import parse_given

    parse_given()

    # set up log output
    import logging

    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("cargo.labor.storage").setLevel(logging.NOTE)

    # print run
    with SQL_Engines.default:
        from contextlib  import closing
        from utexas.data import (
            ResearchSession,
            research_connect,
            )

        ResearchSession.configure(bind = research_connect())

        with closing(ResearchSession()) as session:
            from uuid import UUID

            print_solver_run(session, UUID(module_flags.given.run_uuid))

