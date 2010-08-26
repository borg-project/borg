"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac               import call
    from borg.tools.analyze import main

    call(main)

from plac      import annotations
from cargo.log import get_logger
from borg      import defaults

log = get_logger(__name__, default_level = "INFO")

def commit_features(instance_path, domain, features, url):
    """
    Add feature information to the database.
    """

    # hash the instance
    from borg.tasks import get_task_file_hash

    task_hash = get_task_file_hash(instance_path, domain)

    log.info("instance has hash %s", task_hash.encode("hex_codec"))

    # connect to the database
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default.make_session(url)() as session:
		# locate the instance row
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

@annotations(
    commit = ("commit features to database", "flag"   , "c"),
    url    = ("research database URL"      , "option"),
    )
def main(domain_name, path, commit = False, url = defaults.research_url):
    """
    Acquire task feature information.
    """

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "WARNING")
    get_logger("borg.analyzers",    level = "DETAIL")

    # analyze the instance
    from os.path        import basename
    from cargo.io       import files_under
    from borg.tasks     import (
        FileTask,
        builtin_domains,
        )
    from borg.analyzers import (
        SATzillaAnalyzer,
        UncompressingAnalyzer,
        )

    analyzer = UncompressingAnalyzer(SATzillaAnalyzer())
    domain   = builtin_domains[domain_name]

    for task_path in files_under(path, domain.patterns):
        task     = FileTask(task_path)
        features = analyzer.analyze(task, None)

        log.info("feature pairs follow for %s:", basename(task_path))

        for (name, value) in features.items():
            log.info("%s: %s", name, value)

        # store it, if requested
        if commit:
            commit_features(task_path, domain, features, url)

