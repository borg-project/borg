"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import (
    abstractmethod,
    abstractproperty,
    )
from collections import namedtuple
from cargo.log   import get_logger
from cargo.sugar import ABC
from borg.rowed  import Rowed

log = get_logger(__name__)

Feature = namedtuple("Feature", ["name", "value_type"])

class Feature(Rowed):
    """
    Description of a feature.

    Names are assumed to uniquely identify features.
    """

    def __init__(self, name, value_type, row = None):
        """
        Initialize.
        """

        Rowed.__init__(self, row = row)

        self._name       = name
        self._value_type = value_type

    @property
    def name(self):
        """
        The name of this feature.
        """

        return self._name

    @property
    def value_type(self):
        """
        The type of instances of this feature.
        """

        return self._value_type

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        from borg.data import FeatureRow as FR

        row = session.query(FR).get(self._name)

        if row is None:
            row = FR(name = self._name, type = self._value_type.__name__)
        else:
            if row.type != self._value_type.__name__:
                raise RuntimeError("stored value type does not match feature value type")

        return row

class Analyzer(ABC):
    """
    Abstract base for task feature acquisition classes.
    """

    @abstractmethod
    def analyze(self, task, environment):
        """
        Acquire features of the specified task.

        @return: Mapping from feature names to feature values.
        """

    @abstractproperty
    def features(self):
        """
        Return the features provided by this analyzer.
        """

class NoAnalyzer(Analyzer):
    """
    Acquire no features.
    """

    def analyze(self, task, environment):
        """
        Acquire no features from the specified task.
        """

        return {}

    def get_training(self, session, feature, tasks):
        """
        Return a tasks-by-outcomes array.
        """

        # get stored feature values
        from sqlalchemy import (
            and_,
            select,
            )
        from borg.data  import (
            TaskRow             as TR,
            TaskFloatFeatureRow as TFFR,
            )

        task_uuids = [t.get_row(session).uuid for t in tasks]
        rows       =                                    \
            session.execute(
                select(
                    [TFFR.task_uuid, TFFR.value],
                    and_(
                        TFFR.name == feature.name,
                        TFFR.task_uuid.in_(task_uuids),
                        ),
                    ),
                )                                       \
                .fetchall()

        log.detail("fetched %i rows for feature %s", len(rows), feature.description)

        # build the array
        import numpy

        mapped = dict(rows)
        counts = {
            True  : [0, 1],
            False : [1, 0],
            None  : [0, 0],
            }

        return numpy.array([counts[mapped.get(u)] for u in task_uuids], numpy.uint)

    @property
    def features(self):
        """
        Return the features provided by this analyzer.
        """

        return []

class UncompressingAnalyzer(Analyzer):
    """
    Acquire no features.
    """

    def __init__(self, analyzer):
        """
        Initialize.
        """

        self._analyzer = analyzer

    def analyze(self, task, environment):
        """
        Acquire features from the specified task.
        """

        from borg.tasks import uncompressed_task

        with uncompressed_task(task) as inner_task:
            return self._analyzer.analyze(inner_task, environment)

    @property
    def features(self):
        """
        Return the features provided by this analyzer.
        """

        return self._analyzer.features

class SATzillaAnalyzer(Analyzer):
    """
    Acquire features using SATzilla's (old) analyzer.
    """

    _feature_names = [
        "nvars",
        "nclauses",
        "vars-clauses-ratio",
        "VCG-VAR-mean",
        "VCG-VAR-coeff-variation",
        "VCG-VAR-min",
        "VCG-VAR-max",
        "VCG-VAR-entropy",
        "VCG-CLAUSE-mean",
        "VCG-CLAUSE-coeff-variation",
        "VCG-CLAUSE-min",
        "VCG-CLAUSE-max",
        "VCG-CLAUSE-entropy",
        "POSNEG-RATIO-CLAUSE-mean",
        "POSNEG-RATIO-CLAUSE-coeff-variation",
        "POSNEG-RATIO-CLAUSE-min",
        "POSNEG-RATIO-CLAUSE-max",
        "POSNEG-RATIO-CLAUSE-entropy",
        "POSNEG-RATIO-VAR-mean",
        "POSNEG-RATIO-VAR-stdev",
        "POSNEG-RATIO-VAR-min",
        "POSNEG-RATIO-VAR-max",
        "POSNEG-RATIO-VAR-entropy",
        "UNARY",
        "BINARY+",
        "TRINARY+",
        "HORNY-VAR-mean",
        "HORNY-VAR-coeff-variation",
        "HORNY-VAR-min",
        "HORNY-VAR-max",
        "HORNY-VAR-entropy",
        "horn-clauses-fraction",
        "VG-mean",
        "VG-coeff-variation",
        "VG-min",
        "VG-max",
        "KLB-featuretime",
        "CG-mean",
        "CG-coeff-variation",
        "CG-min",
        "CG-max",
        "CG-entropy",
        "cluster-coeff-mean",
        "cluster-coeff-coeff-variation",
        "cluster-coeff-min",
        "cluster-coeff-max",
        "cluster-coeff-entropy",
        "CG-featuretime",
        ]

    def __init__(self):
        """
        Initialize.
        """

    def analyze(self, task, environment):
        """
        Acquire features of the specified task.
        """

        # sanity
        from borg.tasks import AbstractFileTask

        assert isinstance(task, AbstractFileTask)

        # find the associated feature computation binary
        from borg import get_support_path

        features1s = get_support_path("features1s")

        # execute the helper
        from cargo.io import check_call_capturing

        log.detail("executing %s %s", features1s, task.path)

        (output, _)     = check_call_capturing([features1s, task.path])
        (names, values) = [l.split(",") for l in output.splitlines()]

        if names != self._feature_names:
            raise RuntimeError("unexpected or missing feature names from features1s")
        else:
            return dict(zip([f.name for f in self.features], map(float, values)))

    def get_training(self, session, feature, tasks):
        """
        Return a tasks-by-outcomes array.
        """

        return RecyclingAnalyzer.get_training(session, feature, tasks)

    @property
    def features(self):
        """
        Return the features provided by this analyzer.
        """

        return [Feature("satzilla/%s" % k.lower(), float) for k in self._feature_names]

class BinarySATzillaAnalyzer(Analyzer):
    """
    Acquire features using SATzilla's (old) analyzer.
    """

    def __init__(self, analyzer = SATzillaAnalyzer()):
        """
        Initialize.
        """

        self._analyzer = analyzer

    def analyze(self, task, environment):
        """
        Acquire features of the specified task.
        """

        analysis = self._analyzer.analyze(task, environment)

        return {
            "satzilla/vars-clauses-ratio<=1/4.36" : analysis["satzilla/vars-clauses-ratio"] <= (1 / 4.36),
            }

    def get_training(self, session, feature, tasks):
        """
        Return a tasks-by-outcomes array.
        """

        # sanity
        if feature.name != self.features[0].name:
            raise ValueError("feature not appropriate for this analyzer")

        # get stored feature values
        from sqlalchemy import (
            and_,
            select,
            )
        from borg.data  import (
            TaskRow             as TR,
            TaskFloatFeatureRow as TFFR,
            )

        task_uuids = [t.get_row(session).uuid for t in tasks]
        rows       =                                    \
            session.execute(
                select(
                    [TFFR.task_uuid, TFFR.value],
                    and_(
                        TFFR.name == "satzilla/vars-clauses-ratio",
                        TFFR.task_uuid.in_(task_uuids),
                        ),
                    ),
                )                                       \
                .fetchall()

        if len(rows) != len(tasks):
            log.warning(
                "fetched only %i rows for feature %s of %i tasks",
                 len(rows),
                 len(tasks),
                 feature.name,
                 )
        else:
            log.detail("fetched %i rows for feature %s", len(rows), feature.name)

        # build the array
        import numpy

        mapped = dict(rows)

        def outcome_for(uuid):
            value = mapped.get(uuid)

            if value is None:
                return [0, 0]
            elif value > 1.0 / 4.36:
                return [0, 1]
            else:
                return [1, 0]

        return numpy.array(map(outcome_for, task_uuids), numpy.uint)

    @property
    def features(self):
        """
        Return the names of features provided by this analyzer.
        """

        return [Feature("satzilla/vars-clauses-ratio<=1/4.36", float)]

class RecyclingAnalyzer(Analyzer):
    """
    Look up precomputed features in a database.
    """

    def __init__(self, features):
        """
        Initialize.
        """

        self._features = features

    def analyze(self, task, environment):
        """
        Acquire features of the specified task.
        """

        # sanity
        from borg.tasks import AbstractTask

        assert isinstance(task, AbstractTask)

        # look up the features
        with environment.CacheSession() as session:
            from borg.data import TaskFloatFeatureRow as TFFR

            feature_rows =                                               \
                session                                                  \
                .query(TFFR.name, TFFR.value)                            \
                .filter(TFFR.task == task.get_row(session))              \
                .filter(TFFR.name.in_([f.name for f in self._features])) \
                .all()

            if len(feature_rows) != len(self._features):
                raise RuntimeError("database lacks expected features")

            return dict(feature_rows)

    def get_training(self, session, feature, tasks):
        """
        Return a tasks-by-outcomes array.
        """

        raise NotImplementedError()

    @property
    def features(self):
        """
        Return the features provided by this analyzer.
        """

        return self._features

