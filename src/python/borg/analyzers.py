"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc         import (
    abstractmethod,
    abstractproperty,
    )
from collections import namedtuple
from cargo.log   import get_logger
from cargo.sugar import ABC
from borg.rowed  import Rowed

log = get_logger(__name__)

class Feature(Rowed):
    """
    Description of a feature.

    Names are assumed to uniquely identify features.
    """

    def __init__(self, name, value_type, dimensionality = None, row = None):
        """
        Initialize.
        """

        Rowed.__init__(self, row = row)

        self._name           = name
        self._value_type     = value_type
        self._dimensionality = dimensionality

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

    @property
    def dimensionality(self):
        """
        The feature dimensionality.
        """

        return self._dimensionality

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

class SubsetAnalyzer(Analyzer):
    """
    Acquire no features.
    """

    def __init__(self, analyzer, names = None):
        """
        Initialize.
        """

        self._analyzer = analyzer

        if names is None:
            self._names = [f.name for f in analyzer.features]
        else:
            self._names = set(names)

    def analyze(self, task, environment):
        """
        Acquire features from the specified task.
        """

        analysis = self._analyzer.analyze(task, environment)

        return dict((k, v) for (k, v) in analysis if k in self._names)

    @property
    def features(self):
        """
        Return the features provided by this analyzer.
        """

        return [f for f in self._analyzer.features if f.name in self._names]

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
                raise RuntimeError("database does not contain values for expected features")

            return dict(feature_rows)

    @property
    def features(self):
        """
        Return the features provided by this analyzer.
        """

        return self._features

class BinningAnalyzer(Analyzer):
    """
    Generate histogram features.
    """

    def __init__(self, analyzer, bins):
        """
        Initialize.

        @param analyzer : The raw continuous-feature analyzer.
        @param bins     : A map from feature names to a bin border array.
        """

        self._analyzer = analyzer
        self._bins     = bins
        self._names    = \
            dict(
                zip(
                    (f.name for f in self.features),
                    (f.name for f in analyzer.features),
                    ),
                )

    def analyze(self, task, environment):
        """
        Acquire features of the specified task.
        """

        analysis = self._analyzer.analyze(task, environment)

        def for_value((name, value)):
            ((i,),) = numpy.nonzero(numpy.histogram(value, self._bins[name]))

            return i

        return dict(zip(analysis, map(for_value, analysis.items())))

    def get_training(self, session, feature, task_table):
        """
        Return a tasks-by-outcomes array.
        """

        # get stored feature values
        from sqlalchemy import (
            and_,
            select,
            )
        from borg.data  import TaskFloatFeatureRow as TFFR

        name = self._names[feature.name]
        rows =                                               \
            session.execute(
                select(
                    [TFFR.value],
                    and_(
                        TFFR.name      == name,
                        TFFR.task_uuid == task_table.c.uuid,
                        ),
                    order_by = task_table.c.uuid,
                    ),
                )                                            \
                .fetchall()

        # build the array
        def for_value((value,)):
            (h, _) = numpy.histogram(value, self._bins[name])

            return h

        return numpy.array(map(for_value, rows), numpy.uint)

    @property
    def features(self):
        """
        Return the names of features provided by this analyzer.
        """

        return [
            Feature("%s-%s" % (f.name, self._bins[f.name]), int, self._bins[f.name].size)
            for f in self._analyzer.features
            ]

    @staticmethod
    def _spaced_bin_entry_for(session, task_table, feature, d):
        """
        Select equally-spaced bin positions for a feature.
        """

        from sqlalchemy import (
            func,
            and_,
            select,
            )
        from borg.data  import TaskFloatFeatureRow as TFFR

        ((min_value, max_value),) =                          \
            session.execute(
                select(
                    [
                        func.min(TFFR.value),
                        func.max(TFFR.value),
                        ],
                    and_(
                        TFFR.name      == feature.name,
                        TFFR.task_uuid == task_table.c.uuid,
                        ),
                    ),
                )

        edges     =  numpy.r_[min_value:max_value:(d + 1) * 1j]
        edges[ 0] = -numpy.inf
        edges[-1] =  numpy.inf

        return (feature.name, edges)

    @staticmethod
    def _percentile_bin_entry_for(session, task_table, feature, d):
        """
        Select percentile-spaced bin positions for a feature.
        """

        from sqlalchemy import (
            func,
            select,
            )
        from borg.data  import TaskFloatFeatureRow as TFFR

        where = (TFFR.name == feature.name) & (TFFR.task_uuid == task_table.c.uuid)
        size  = session.execute(select([func.count()], where)).scalar()

        def value_at(p):
            return                                      \
                session.execute(
                    select(
                        [TFFR.value],
                        where,
                        limit    = 1,
                        offset   = int((size - 1) * p),
                        order_by = TFFR.value,
                        ),
                    )                                   \
                    .scalar()

        edges = [-numpy.inf] + map(value_at, numpy.r_[0.0:1.0:(d + 1) * 1j][1:-1]) + [numpy.inf]

        return (feature.name, numpy.asarray(edges))

    @staticmethod
    def _build(trainer, raw_analyzer, d, get_bin_entries):
        """
        Return an analyzer with bins fit to training data.
        """

        with trainer.context() as (session, task_table):
            bin_entry_for = lambda f: get_bin_entries(session, task_table, f, d)
            bins          = dict(map(bin_entry_for, raw_analyzer.features))

            for (name, edges) in bins.items():
                log.detail("bins for %s: %s", name, edges)

            return BinningAnalyzer(raw_analyzer, bins)

    @staticmethod
    def spaced(trainer, raw_analyzer, d):
        """
        Return an analyzer with bins fit to training data.
        """

        get_bin_entries = BinningAnalyzer._spaced_bin_entry_for

        return BinningAnalyzer._build(trainer, raw_analyzer, d, get_bin_entries)

    @staticmethod
    def percentile(trainer, raw_analyzer, d):
        """
        Return an analyzer with bins fit to training data.
        """

        get_bin_entries = BinningAnalyzer._percentile_bin_entry_for

        return BinningAnalyzer._build(trainer, raw_analyzer, d, get_bin_entries)

