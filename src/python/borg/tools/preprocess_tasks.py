"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.preprocess_tasks import main

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
            "--restarts",
            type    = int,
            default = 1,
            metavar = "INT",
            help    = "make INT restarts per task [%default]",
            ),
        )

def preprocess_task(
    engine_url,
    preprocessor,
    trial_row,
    input_task_row,
    restarts,
    named_solvers,
    budget,
    tasks_path,
    output_path,
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

        environment = \
            Environment(
                MainSession   = MainSession,
                named_solvers = named_solvers,
                )

        # merge unpickled rows
        trial_row      = session.merge(trial_row)
        input_task_row = session.merge(input_task_row)

        session.commit()

        # make the preprocessor runs
        from os.path    import join
        from cargo.io   import mkdtemp_scoped
        from borg.tasks import FileTask

        for i in xrange(restarts):
            # prepare this run
            input_task_name_row = input_task_row.names[0]
            input_task_path     = join(tasks_path, input_task_name_row.name)
            input_task          = FileTask(input_task_path, row = input_task_row)

            log.info(
                "preprocessing %s (run %i of %i)",
                input_task_name_row.name,
                i + 1,
                restarts,
                )

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

                log.info("run_row %s", run_row)
                log.info("run_row uuid %s", run_row.uuid)

                # save the preprocessed task data, if necessary
                if attempt.output_task != attempt.task:
                    from os      import rename
                    from os.path import lexists
                    from shutil  import copytree

                    output_task_row = attempt.output_task.get_row(session)
                    final_path      = join(output_path, output_task_row.uuid.hex)

                    if not lexists(final_path):
                        stage_path = "%s.partial" % final_path

                        copytree(temporary_path, stage_path)
                        rename(stage_path, final_path)

            # flush this row to the database
            session.commit()

def yield_jobs(session, preprocessor_name, budget, tasks_path, output_path, prefix):
    """
    Generate a set of jobs to distribute.
    """

    # create a trial
    from cargo.temporal import utc_now
    from borg.data      import TrialRow

    trial_row = TrialRow(label = "preprocessor runs (at %s)" % utc_now())

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

    task_rows     = TaskRow.with_prefix(session, prefix)
    named_solvers = get_named_solvers()
    preprocessor  = UncompressingPreprocessor(LookupPreprocessor(preprocessor_name))
    restarts      = module_flags.given.restarts

    for task_row in task_rows:
        yield CallableJob(
            preprocess_task,
            engine_url     = session.connection().engine.url,
            preprocessor   = preprocessor,
            trial_row      = trial_row,
            input_task_row = task_row,
            restarts       = restarts,
            named_solvers  = named_solvers,
            budget         = budget,
            tasks_path     = tasks_path,
            output_path    = output_path,
            )

def main():
    """
    Run the script.
    """

    # get command line arguments
    import cargo.labor.storage
    import borg.data
    import borg.solvers

    from cargo.flags    import parse_given
    from cargo.temporal import TimeDelta

    (preprocessor_name, budget, tasks_path, output_path, prefix) = \
        parse_given(
            usage = "%prog <preprocessor> <budget> <tasks> <output> <prefix> [options]",
            )

    budget = TimeDelta(seconds = float(budget))

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "DETAIL")

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
                    yield_jobs(
                        session,
                        preprocessor_name,
                        budget,
                        abspath(tasks_path),
                        abspath(output_path),
                        prefix,
                        ),
                    )

        # run the jobs
        from cargo.labor.storage import outsource_or_run
        from cargo.temporal      import utc_now

        outsource_or_run(jobs, "preprocessing %s (at %s)" % (prefix, utc_now()))

