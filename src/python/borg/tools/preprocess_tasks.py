"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                        import call
    from borg.tools.preprocess_tasks import main

    call(main)

from plac           import annotations
from cargo.log      import get_logger
from cargo.json     import load_json
from cargo.temporal import parse_timedelta

log = get_logger(__name__, default_level = "INFO")

def preprocess_task(
    engine_url,
    preprocessor,
    trial_row,
    input_task_uuid,
    restarts,
    named_solvers,
    budget,
    collections,
    ):
    """
    Make some number of preprocessor runs on a task.
    """

    # make sure that we're logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("borg.solvers.satelite", level = "DEBUG")

    # connect to the database
    from cargo.sql.alchemy import (
        SQL_Engines,
        make_session,
        )

    main_engine = SQL_Engines.default.get(engine_url)
    MainSession = make_session(bind = main_engine)

    with MainSession() as session:
        # set up the environment
        from borg.solvers import Environment

        output_path = collections[None]
        environment = \
            Environment(
                MainSession   = MainSession,
                named_solvers = named_solvers,
                collections   = collections,
                )

        # merge unpickled rows
        from borg.data import TaskRow

        trial_row      = session.merge(trial_row)
        input_task_row = session.query(TaskRow).get(input_task_uuid)

        if input_task_row is None:
            raise RuntimeError("no such task")

        # make the preprocessor runs
        from cargo.io import mkdtemp_scoped

        for i in xrange(restarts):
            # prepare this run
            input_task = input_task_row.get_task(environment)

            log.info("preprocessing %s (run %i of %i)", input_task_uuid, i + 1, restarts)

            with mkdtemp_scoped() as temporary_path:
                # make this run
                attempt = \
                    preprocessor.preprocess(
                        input_task,
                        budget,
                        temporary_path,
                        None,
                        environment,
                        )

                # prepare to store the attempt
                run_row = attempt.get_row(session)

                run_row.trials = [trial_row]

                # save the preprocessed task data, if necessary
                if attempt.output_task != attempt.task:
                    from os      import rename
                    from os.path import (
                        join,
                        lexists,
                        )
                    from shutil  import copytree

                    output_task_row = attempt.output_task.get_row(session)
                    final_path      = join(output_path, output_task_row.uuid.hex)

                    log.info("output task is %s", output_task_row)

                    if not lexists(final_path):
                        stage_path = "%s.partial" % final_path

                        copytree(temporary_path, stage_path)
                        rename(stage_path, final_path)

                        log.note("copied output from %s to %s", temporary_path, final_path)

            # flush this row to the database
            session.commit()

def yield_jobs(session, preprocessor_names, budget, task_uuids, collections, restarts):
    """
    Generate a set of jobs to distribute.
    """

    # create a trial
    from cargo.temporal import utc_now
    from borg.data      import TrialRow

    trial_row = TrialRow(label = "preprocessing tasks (at %s)" % utc_now())

    session.add(trial_row)
    session.commit()

    # yield jobs
    from cargo.labor.jobs import CallableJob
    from borg.data        import TaskRow
    from borg.solvers     import (
        LookupPreprocessor,
        UncompressingPreprocessor,
        get_named_solvers,
        )

    named_solvers = get_named_solvers()

    for preprocessor_name in preprocessor_names:
        preprocessor = UncompressingPreprocessor(LookupPreprocessor(preprocessor_name))

        for task_uuid in task_uuids:
            yield CallableJob(
                preprocess_task,
                engine_url      = session.connection().engine.url,
                preprocessor    = preprocessor,
                trial_row       = trial_row,
                input_task_uuid = task_uuid,
                restarts        = restarts,
                named_solvers   = named_solvers,
                budget          = budget,
                collections     = collections,
                )

@annotations(
    budget    = ("solver budget"    , "positional", None, parse_timedelta),
    arguments = ("arguments in JSON", "positional", None, load_json)   ,
    restarts  = ("restarts per task", "option"    , "r" , int)         ,
    )
def main(budget, arguments, restarts = 1):
    """
    Run the script.
    """

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "WARNING")

    # connect to the database and go
    from cargo.sql.alchemy import (
        SQL_Engines,
        make_session,
        )

    with SQL_Engines.default:
        from os.path   import abspath
        from borg.data import research_connect

        ResearchSession = make_session(bind = research_connect())

        # build jobs
        from uuid       import UUID
        from borg.tasks import get_collections

        with ResearchSession() as session:
            jobs = \
                list(
                    yield_jobs(
                        session,
                        arguments["preprocessors"],
                        budget,
                        map(UUID, arguments["tasks"]),
                        get_collections(),
                        restarts,
                        ),
                    )

        # run the jobs
        from cargo.labor.storage import outsource_or_run
        from cargo.temporal      import utc_now

        outsource_or_run(jobs, "preprocessing tasks (at %s)" % utc_now())

