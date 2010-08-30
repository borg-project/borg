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

    def __init__(self):
        """
        Initialize.
        """

        self._analyzer = SATzillaAnalyzer()

    def analyze(self, task, environment):
        """
        Acquire features of the specified task.
        """

        analysis = self._analyzer.analyze(task, environment)

        return {
            "satzilla/vars-clauses-ratio<=1/4.36" : analysis["satzilla/vars-clauses-ratio"] <= (1 / 4.36),
            }

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

    def __init__(self, names):
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
            from borg.data import TaskFeatureRow as TFR

            constraint = TFR.task == task.get_row(session)

            if self._names is not None:
                constraint = constraint & TFR.name.in_(self._names)

            feature_rows = session.query(TFR.name, TFR.value).filter(constraint).all()

            return dict(feature_rows)

    @property
    def features(self):
        """
        Return the names of features provided by this analyzer.
        """

        return self._names

