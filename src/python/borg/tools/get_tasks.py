"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                 import call
    from borg.tools.get_tasks import main

    call(main)

from plac        import annotations
from cargo.log   import get_logger
from cargo.sugar import composed
from borg        import defaults

log = get_logger(__name__, default_level = "INFO")

class GetTaskJob(object):
    """
    Add a task.
    """

    def __init__(self, engine_url, task_path, name, collection, domain):
        """
        Initialize.
        """

        self._engine_url = engine_url
        self._task_path  = task_path
        self._name       = name
        self._collection = collection
        self._domain     = domain

    def __call__(self):
        """
        Run the job.
        """

        # make sure that we're logging
        from cargo.log import enable_default_logging

        enable_default_logging()

        # connect to the research database
        from cargo.sql.alchemy import (
            SQL_Engines,
            make_session,
            )

        ResearchSession = make_session(bind = SQL_Engines.default.get(self._engine_url))

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
                        TNR.name       == self._name,
                        TNR.collection == self._collection,
                    ),
                )                                     \
                .first()

            if task_name_row is not None:
                log.note("database already contains %s", self._name)

                return

            # find or create the task row
            from cargo.sql.alchemy import lock_table
            from borg.data         import FileTaskRow as FTR
            from borg.tasks        import get_task_file_hash

            lock_table(session.connection(), FTR.__tablename__, "share row exclusive")

            file_hash = get_task_file_hash(self._task_path, self._domain)
            task_row  = session.query(FTR).filter(FTR.hash == buffer(file_hash)).first()

            if task_row is None:
                task_row = FTR(hash = buffer(file_hash))

                session.add(task_row)

            session.commit()

            # create the task name row
            task_name_row = TNR(task = task_row, name = self._name, collection = self._collection)

            session.add(task_name_row)
            session.commit()

            # tell the world
            log.info("added task %s with hash %s", task_row.uuid, file_hash.encode("hex_codec"))

@composed(list)
def get_task_jobs(session, tasks_path, relative_to, collection, domain_name):
    """
    Find tasks to hash and name.
    """

    # build tasks
    from os.path    import relpath
    from cargo.io   import files_under
    from borg.tasks import builtin_domains

    domain = builtin_domains[domain_name]

    for task_path in files_under(tasks_path, domain.patterns):
        yield GetTaskJob(
            engine_url = session.connection().engine.url,
            task_path  = task_path,
            name       = relpath(task_path, relative_to),
            collection = collection,
            domain     = domain,
            )

@annotations(
    tasks_path  = ("find tasks under", )        ,
    relative_to = ("collection root" , )        ,
    domain      = ("problem domain"  , "option" , "d" , str, ["sat", "pb"]),
    collection  = ("task name group" , "option" , "c"),
    url         = ("database URL"    , "option"),
    )
def main(
    tasks_path,
    relative_to,
    domain     = "sat",
    collection = "default",
    url        = defaults.research_url,
    ):
    """
    Run the script.
    """

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    # connect to the database and go
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default as engines:
        with engines.make_session(url)() as session:
            from os.path import abspath

            jobs = \
                list(
                    get_task_jobs(
                        session,
                        abspath(tasks_path),
                        abspath(relative_to),
                        collection,
                        domain,
                        ),
                    )

        # run the jobs
        from cargo.labor    import outsource_or_run
        from cargo.temporal import utc_now

        outsource_or_run(jobs, False, "adding task rows (at %s)" % utc_now())

