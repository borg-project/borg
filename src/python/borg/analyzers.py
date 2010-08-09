"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import abstractmethod
from cargo.log   import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

class TaskAnalyzer(ABC):
    """
    Abstract base for task feature acquisition classes.
    """

    @abstractmethod
    def analyze(self, task, environment):
        """
        Acquire features of the specified task.

        @return: Mapping from feature names to feature values.
        """

    @staticmethod
    def build(request, trainer):
        """
        Build an analyzer as specified.
        """

        builders = {
            "no"        : NoAnalyzer.build,
            "satzilla"  : SATzillaAnalyzer.build,
            "recycling" : RecyclingAnalyzer.build,
            }

        return builders[request["type"]](request, trainer)

class NoAnalyzer(TaskAnalyzer):
    """
    Acquire no features.
    """

    def analyze(self, task, environment):
        """
        Acquire no features from the specified task.
        """

        return {}

    @staticmethod
    def build(request, trainer):
        """
        Build this analyzer from a request.
        """

        return NoAnalyzer()

class SATzillaAnalyzer(TaskAnalyzer):
    """
    Acquire features using SATzilla's (old) analyzer.
    """

    def __init__(self, names = None):
        """
        Initialize.
        """

        self._names = names

    def analyze(self, task, environment):
        """
        Acquire features of the specified task.
        """

        # sanity
        from borg.tasks import AbstractFileTask

        assert isinstance(task, AbstractFileTask)

        # compute the features
        from borg.sat.cnf import compute_raw_features

        raw         = compute_raw_features(task.path)
        transformed = {
            "satzilla/vars-clauses-ratio<=4.36" : raw["vars-clauses-ratio"] <= 4.36,
            }

        if self._names is None:
            return transformed
        else:
            return dict((n, transformed[n]) for n in self._names)

    @staticmethod
    def build(request, trainer):
        """
        Build this analyzer from a request.
        """

        return SATzillaAnalyzer(request.get("names"))

class RecyclingAnalyzer(TaskAnalyzer):
    """
    Look up precomputed features from the database.
    """

    def __init__(self, names = None):
        """
        Initialize.
        """

        self._names = names

    def analyze(self, task, environment):
        """
        Acquire features of the specified task.
        """

        # sanity
        from borg.tasks import AbstractTask

        assert isinstance(task, AbstractTask)

        # look up the features
        with environment.CacheSession() as session:
            from borg.data  import TaskFeatureRow as TFR

            constraint = TFR.task == task_row

            if self._names is not None:
                constraint = constraint & TFR.name.in_(self._names)

            task_row     = task.get_row(session)
            feature_rows = session.query([TFR.name, TFR.value]).filter(constraint).all()

            return dict(feature_rows)

    @staticmethod
    def build(request, trainer):
        """
        Build this analyzer from a request.
        """

        return RecyclingAnalyzer(request.get("names"))

