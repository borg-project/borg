"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.get_tasks import main

    raise SystemExit(main())

from cargo.log   import get_logger
from cargo.flags import (
    Flag,
    Flags,
    )

log          = get_logger(__name__, default_level = "NOTE")
module_flags = \
    Flags(
        "Script Options",
        Flag(
            "--collection",
            default = "default",
            metavar = "NAME",
            help    = "task names in collection NAME [%default]",
            ),
        Flag(
            "-d",
            "--domain",
            type    = "choice",
            choices = ["sat", "pb"],
            default = "sat",
            metavar = "NAME",
            help    = "find NAME tasks [%default]",
            ),
        )

def get_task(
    engine_url,
    task_path,
    name,
    collection,
    domain,
    ):
    """
    Add a task.
    """

    # make sure that we're logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    # connect to the research database
    from cargo.sql.alchemy import (
        SQL_Engines,
        make_session,
        )

    ResearchSession = make_session(bind = SQL_Engines.default.get(engine_url))

    # get the task
    with ResearchSession() as session:
        # has this task already been stored?
        from sqlalchemy import and_
        from borg.data  import TaskNameRow as TNR

        task_name_row =                           \
            session                               \
            .query(TNR)                           \
            .filter(
                and_(
                    TNR.name       == name,
                    TNR.collection == collection,
                ),
            )                                     \
            .first()

        if task_name_row is not None:
            log.note("database already contains %s", name)

            return

        # find or create the task row
        from cargo.sql.alchemy import lock_table
        from borg.data         import FileTaskRow as FTR
        from borg.tasks        import get_task_file_hash

        lock_table(session.connection(), FTR.__tablename__, "share row exclusive")

        file_hash = get_task_file_hash(task_path, domain)
        task_row  = session.query(FTR).filter(FTR.hash == buffer(file_hash)).first()

        if task_row is None:
            task_row = FTR(hash = buffer(file_hash))

            session.add(task_row)

        session.commit()

        # create the task name row
        task_name_row = TNR(task = task_row, name = name, collection = collection)

        session.add(task_name_row)
        session.commit()

        # tell the world
        log.info("added task %s with hash %s", task_row.uuid, file_hash.encode("hex_codec"))

def yield_get_task_jobs(session, tasks_path, relative_to, collection, domain_name):
    """
    Find tasks to hash and name.
    """

    # build tasks
    from os.path          import relpath
    from cargo.io         import files_under
    from cargo.labor.jobs import CallableJob
    from borg.tasks       import builtin_domains

    domain = builtin_domains[domain_name]

    for task_path in files_under(tasks_path, domain.patterns):
        yield CallableJob(
            get_task,
            engine_url = session.connection().engine.url,
            task_path  = task_path,
            name       = relpath(task_path, relative_to),
            collection = collection,
            domain     = domain,
            )

def main():
    """
    Run the script.
    """

    # get command line arguments
    import cargo.labor.storage
    import borg.data

    from cargo.flags import parse_given

    (tasks_path, relative_to) = parse_given(usage = "%prog <tasks> <relative> [options]")

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    # connect to the database and go
    from cargo.sql.alchemy import (
        SQL_Engines,
        make_session,
        )

    with SQL_Engines.default:
        from os.path   import abspath
        from borg.data import research_connect

        ResearchSession = make_session(bind = research_connect())

        with ResearchSession() as session:
            jobs = \
                list(
                    yield_get_task_jobs(
                        session,
                        abspath(tasks_path),
                        abspath(relative_to),
                        module_flags.given.collection,
                        module_flags.given.domain,
                        ),
                    )

        # run the jobs
        from cargo.labor.storage import outsource_or_run
        from cargo.temporal      import utc_now

        outsource_or_run(jobs, "adding task rows (at %s)" % utc_now())

