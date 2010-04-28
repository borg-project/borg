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

log          = get_logger(__name__, default_level = "NOTSET")
module_flags = \
    Flags(
        "Run Printing",
        Flag(
            "--run-uuid",
            metavar = "UUID",
            help    = "print run UUID [%default]",
            ),
        )

def print_run(session, run_uuid):
    """
    Print a specific CPU-limited run.
    """

    from utexas.data import CPU_LimitedRunRow

    runs =                                          \
        session                                     \
        .query(CPU_LimitedRunRow)                   \
        .filter(CPU_LimitedRunRow.uuid == run_uuid)

    for run in runs:
        log.info("CPU-LIMITED RUN: %s", run.uuid)
        log.info("started: %s", run.started)
        log.info("usage_elapsed: %s", run.usage_elapsed)
        log.info("proc_elapsed: %s", run.proc_elapsed)
        log.info("cutoff: %s", run.cutoff)
        log.info("fqdn: %s", run.fqdn)
        log.info("exit_status: %s", run.exit_status)
        log.info("exit_signal: %s", run.exit_signal)

        stdout = run.get_stdout()

        log.info(
            "stdout follows (%i characters)\n%s",
            len(stdout),
            stdout,
            )

        stderr = run.get_stderr()

        log.info(
            "stderr follows (%i characters)\n%s",
            len(stderr),
            stderr,
            )

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
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("cargo.labor.storage", level = "NOTE")

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

            print_run(session, UUID(module_flags.given.run_uuid))

