"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.sat.get_tasks import main

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
            metavar  = "NAME",
            help    = "task names in collection NAME [%default]",
            ),
        )

def get_task(
    engine_url,
    task_path,
    name,
    collection,
    ):
    """
    Add a task.
    """

    # connect to the research database
    from cargo.sql.alchemy import (
        SQL_Engines,
        make_session,
        )

    ResearchSession = make_session(bind = SQL_Engines.default.get(engine_url))

    # get the task
    with ResearchSession() as session:
        # has this task already been stored?
        from sqlalchemy  import and_
        from utexas.data import TaskNameRow as TN

        task_name_row =                          \
            session                              \
            .query(TN)                           \
            .filter(
                and_(
                    TN.name       == name,
                    TN.collection == collection,
                ),
            )                                    \
            .first()

        if task_name_row is not None:
            log.debug("database already contains %s", name)

            return

        # hash the task file
        from os.path        import join
        from cargo.io       import (
            decompress_if,
            mkdtemp_scoped,
            hash_yielded_bytes,
            )
        from utexas.sat.cnf import yield_sanitized_cnf

        log.debug("hashing %s", name)

        with mkdtemp_scoped(prefix = "cnf.") as sandbox_path:
            uncompressed_path = \
                decompress_if(
                    task_path,
                    join(sandbox_path, "uncompressed.cnf"),
                    )

            with open(uncompressed_path) as file:
                (_, file_hash) = hash_yielded_bytes(yield_sanitized_cnf(file), "sha512")

        # find or create the task row
        from utexas.data import SAT_TaskRow as ST

        task_row = session.query(ST).filter(ST.hash == buffer(file_hash)).first()

        if task_row is None:
            task_row = ST(hash = buffer(file_hash))

            session.add(task_row)

        # create the task name row
        task_name_row = TN(task = task_row, name = name, collection = collection)

        session.add(task_name_row)

        # and we're done
        log.note("stored row for %s", name)

        session.commit()

def yield_get_task_jobs(session, tasks_path, relative_to, collection):
    """
    Find tasks to hash and name.
    """

    from os.path          import (
        join,
        relpath,
        )
    from cargo.io         import files_under
    from cargo.labor.jobs import CallableJob

    patterns = ["*.cnf", "*.cnf.gz", "*.cnf.bz2", "*.cnf.xz"]

    for task_path in files_under(tasks_path, patterns):
        yield CallableJob(
            get_task,
            engine_url = session.connection().engine.url,
            task_path  = task_path,
            name       = relpath(task_path, relative_to),
            collection = collection,
            )

def main():
    """
    Deal with core database metadata.
    """

    # get command line arguments
    import cargo.labor.storage
    import utexas.data

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
        from os.path     import abspath
        from utexas.data import research_connect

        ResearchSession = make_session(bind = research_connect())

        with ResearchSession() as session:
            jobs = \
                list(
                    yield_get_task_jobs(
                        session,
                        abspath(tasks_path),
                        abspath(relative_to),
                        module_flags.given.collection,
                        ))

        # run the jobs
        from cargo.labor.storage import outsource_or_run
        from cargo.temporal      import utc_now

        outsource_or_run(jobs, "making preprocessor runs at %s" % utc_now())

