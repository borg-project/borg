"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log    import get_logger
from borg.solvers import TaskAnalyzer

log = get_logger(__name__)

def transform_satzilla_features(raw):
    """
    Transform raw features into relevant binary features.
    """

    return {
        "satzilla/vars-clauses-ratio<=4.36" : raw["vars-clauses-ratio"] <= 4.36,
        }

class SATzillaAnalyzer(TaskAnalyzer):
    """
    Acquire features using SATzilla's (old) analyzer.
    """

    def analyze(self, task, environment):
        """
        Acquire features of the specified task.
        """

        # sanity
        from borg.tasks import AbstractFileTask

        assert isinstance(task, AbstractFileTask)

        # compute the features
        from borg.sat.cnf import compute_raw_features

        return transform_satzilla_features(compute_raw_features(task.path))

class RecyclingAnalyzer(TaskAnalyzer):
    """
    Look up precomputed features from the database.
    """

    def analyze(self, task, environment):
        """
        Acquire features of the specified task.
        """

        # sanity
        from borg.tasks import AbstractTask

        assert isinstance(task, AbstractTask)

        # look up the features
        with environment.CacheSession() as session:
            from borg.data import TaskFeatureRow as TFR

            task_row     = task.get_row(session)
            feature_rows = session.query(TFR).filter(TFR.task == task_row).all()

            return dict((r.name, r.value) for r in feature_rows)

