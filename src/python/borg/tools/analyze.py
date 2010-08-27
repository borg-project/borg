"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac               import call
    from borg.tools.analyze import main

    call(main)

from plac        import annotations
from cargo.log   import get_logger
from cargo.sugar import composed
from borg        import defaults

log = get_logger(__name__, default_level = "INFO")

class AnalyzeTaskJob(object):
    """
    Analyze a specified task file.
    """

    def __init__(self, analyzer, domain, path, url = None):
        """
        Initialize.
        """

        self._analyzer = analyzer
        self._domain   = domain
        self._path     = path
        self._url      = url

    def __call__(self):
        """
        Run the analysis.
        """

        # log as appropriate
        get_logger("borg.sat.cnf",   level = "DETAIL")
        get_logger("borg.analyzers", level = "DETAIL")

        # analyze the task
        from os.path    import basename
        from borg.tasks import FileTask

        task     = FileTask(self._path)
        features = self._analyzer.analyze(task, None)

        log.info("feature pairs follow for %s:", basename(self._path))

        for (name, value) in features.items():
            log.info("%s: %s", name, value)

        # store the analysis, if requested
        if self._url is not None:
            self.commit(features)

    def commit(self, features):
        """
        Add feature information to the database.
        """

        # hash the instance
        from borg.tasks import get_task_file_hash

        log.info("hashing task file")

        task_hash = get_task_file_hash(self._path, self._domain)

        log.info("instance has hash %s", task_hash.encode("hex_codec"))

        # connect to the database
        from cargo.sql.alchemy import SQL_Engines

        with SQL_Engines.default.make_session(self._url)() as session:
            # look up the instance row
            from borg.data import (
                FileTaskRow    as FTR,
                TaskFeatureRow as TFR,
                )

            task_row = session.query(FTR).filter(FTR.hash == buffer(task_hash)).first()

            if task_row is None:
                raise RuntimeError("cannot locate row corresponding to task")

            log.info("task row has uuid %s", task_row.uuid)

            # then insert corresponding feature rows
            for (name, value) in features.items():
                constraint = (TFR.task == task_row) & (TFR.name == name)

                if session.query(TFR).filter(constraint).scalar() is None:
                    feature_row = TFR(task = task_row, name = name, value = value)

                    session.add(feature_row)
                else:
                    log.info("feature \"%s\" already stored", name)

            session.commit()

    @staticmethod
    @composed(list)
    def for_directory(domain_name, path, url = None):
        """
        Return analysis jobs for tasks under path.
        """

        from cargo.io       import files_under
        from borg.tasks     import builtin_domains
        from borg.analyzers import (
            SATzillaAnalyzer,
            UncompressingAnalyzer,
            )

        analyzer = UncompressingAnalyzer(SATzillaAnalyzer())
        domain   = builtin_domains[domain_name]

        for task_path in files_under(path, domain.patterns):
            yield AnalyzeTaskJob(analyzer, domain, task_path, url)

@annotations(
    commit    = ("commit features to database", "flag"   , "c"),
    url       = ("research database URL"      , "option"),
    outsource = ("outsource jobs"             , "flag")  ,
    )
def main(domain_name, path, commit = False, url = defaults.research_url, outsource = False):
    """
    Acquire task feature information.
    """

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    # run the jobs
    from cargo.labor import outsource_or_run

    jobs = AnalyzeTaskJob.for_directory(domain_name, path, url if commit else None)

    outsource_or_run(jobs, outsource)

