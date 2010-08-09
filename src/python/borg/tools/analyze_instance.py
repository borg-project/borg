"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.analyze_instance import main

    raise SystemExit(main())

from cargo.log   import get_logger
from cargo.flags import (
    Flag,
    Flags,
    )

log          = get_logger(__name__, default_level = "INFO")
module_flags = \
    Flags(
        "Script Options",
        Flag(
            "--commit",
            action = "store_true",
            help   = "commit features to database [%default]",
            ),
        )

def commit_features(instance_path, domain_name, features):
    """
    Add feature information to the database.
    """

    # hash the instance
    from borg.tasks import (
        builtin_domains,
        get_task_file_hash,
        )

    task_hash = get_task_file_hash(instance_path, builtin_domains[domain_name])

    log.info("instance has hash %s", task_hash.encode("hex_codec"))

    # connect to the database
    from cargo.sql.alchemy import (
        SQL_Engines,
        make_session,
        )
    from borg.data         import research_connect

    ResearchSession = make_session(bind = research_connect())

    # locate the instance row
    with ResearchSession() as session:
        from borg.data import (
            FileTaskRow as FTR,
            TaskFeatureRow as TFR,
            )

        task_row = session.query(FTR).filter(FTR.hash == buffer(task_hash)).first()

        if task_row is None:
            raise RuntimeError("cannot locate row corresponding to task")

        log.info("task row has uuid %s", task_row.uuid)

        for (name, value) in features.items():
            constraint = (TFR.task == task_row) & (TFR.name == name)

            if session.query(TFR).filter(constraint).scalar() is None:
                feature_row = TFR(task = task_row, name = name, value = value)

                session.add(feature_row)
            else:
                log.info("feature \"%s\" already stored", name)

        session.commit()

def main():
    """
    Main.
    """

    # get command line arguments
    import borg.data

    from cargo.flags import parse_given

    (instance_path, domain_name) = parse_given(usage = "%prog [options] <instance> <domain>")

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "WARNING")
    get_logger("borg.analyzers",    level = "DETAIL")

    # analyze the instance
    from borg.tasks     import FileTask
    from borg.analyzers import SATzillaAnalyzer

    task     = FileTask(instance_path)
    analyzer = SATzillaAnalyzer()
    features = analyzer.analyze(task, None)

    log.info("feature pairs follow:")

    for (name, value) in features.items():
        log.info("%s: %s", name, value)

    # store it, if requested
    if module_flags.given.commit:
        commit_features(instance_path, domain_name, features)

