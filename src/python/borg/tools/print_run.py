"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                 import call
    from borg.tools.print_run import main

    call(main)

from uuid      import UUID
from plac      import annotations
from cargo.log import get_logger

log = get_logger(__name__, default_level = "NOTSET")

def print_run(session, run_uuid):
    """
    Print a specific CPU-limited run.
    """

    from borg.data import CPU_LimitedRunRow

    run = session.query(CPU_LimitedRunRow).get(run_uuid)

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

@annotations(
    run_uuid = ("run to print", "positional", None, UUID),
    )
def main(run_uuid):
    """
    Application body.
    """

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    # print run
    with SQL_Engines.default:
        from cargo.sql.alchemy import make_session
        from borg.data         import research_connect

        ResearchSession = make_session(bind = research_connect())

        with ResearchSession() as session:
            print_run(session, run_uuid)

