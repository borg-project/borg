"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import sklearn.pipeline
import sklearn.linear_model
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def prepare_training_data_raw(run_data, model):
    """Prepare regression training data from run data."""

    (_, _, C) = model.log_masses.shape
    feature_values_NF = run_data.to_features_array()
    counts = run_data.to_bins_array(run_data.solver_names, C - 1)
    log_ps_NM = borg.models.sampled_pmfs_log_pmf(model.log_masses, counts)
    ps_NM = numpy.exp(log_ps_NM)
    ps_NM /= numpy.sum(ps_NM, axis = -1)[..., None]

    return (feature_values_NF, ps_NM)

def prepare_training_data_logit(run_data, model):
    """Prepare regression training data from run data."""

    (feature_values_NF, ps_NM) = prepare_training_data_raw(run_data, model)

    ps_NM += 1e-5
    ps_NM /= 1.0 + 1e-4

    assert numpy.all(0.0 < ps_NM)
    assert numpy.all(ps_NM < 1.0)

    logit_ps_NM = numpy.log(ps_NM / (1.0 - ps_NM))

    return (feature_values_NF, logit_ps_NM)

class UniformRegression(object):
    """Uninformative regression model."""

    def __init__(self, run_data, model):
        """Initialize."""

        (_, ps_NM) = prepare_training_data_raw(run_data, model)
        (_, self._M) = ps_NM.shape

    def predict(self, task, features, normalize = True):
        """Predict RTD probabilities."""

        return numpy.ones(self._M) / self._M

class OracleRegression(object):
    """Perfect regression."""

    def __init__(self, run_data, model):
        """Initialize."""

        self._run_data = run_data
        (_, self._ps_NM) = prepare_training_data_raw(run_data, model)

    def predict(self, task, features, normalize = True):
        """Predict RTD probabilities."""

        prediction = self._ps_NM[sorted(self._run_data.run_lists).index(task)]

        if normalize:
            prediction /= numpy.sum(prediction)

        assert numpy.sum(prediction) > 0.0
        assert numpy.abs(numpy.sum(prediction) - 1.0) < 1e-4

        return prediction

class ConstantRegression(object):
    """Constant regression model."""

    def __init__(self, run_data, model):
        """Initialize."""

        (features_NF, ps_NM) = prepare_training_data_raw(run_data, model)

        self._prediction_M = numpy.mean(ps_NM, axis = 0)

    def predict(self, task, features, normalize = True):
        """Predict RTD probabilities."""

        prediction_M = numpy.copy(self._prediction_M)

        if normalize:
            prediction_M /= numpy.sum(prediction_M)

        return prediction_M

class LinearRegression(object):
    """Linear regression model."""

    def __init__(self, run_data, model, K = 32):
        """Initialize."""

        # cluster the distributions
        features_NF = run_data.to_features_array()

        self._model = model
        self._kl_means = borg.bregman.KLMeans(k = K).fit(numpy.exp(model.log_masses))
        self._indicators = borg.statistics.indicator(self._kl_means._assignments, K)
        self._regression = \
            sklearn.pipeline.Pipeline([
                ("scaler", sklearn.preprocessing.Scaler()),
                ("estimator", sklearn.linear_model.LogisticRegression(C = 1e-1)),
                ]) \
                .fit(features_NF, self._kl_means._assignments)

        (_, estimator) = self._regression.steps[-1]

        self._indices = dict(zip(estimator.label_, xrange(estimator.label_.shape[0])))
        self._sizes = numpy.sum(self._indicators, axis = 0)

    def predict(self, features):
        """Predict RTD probabilities."""

        features = numpy.asarray(features)

        (M, F) = features.shape
        (N, K) = self._indicators.shape

        prediction = self._regression.predict_proba(features)
        weights = numpy.empty((M, N))

        for m in xrange(M):
            for n in xrange(N):
                k = self._kl_means._assignments[n]
                p_z = prediction[m, self._indices[k]]

                weights[m, n] = p_z / self._sizes[k]

        return weights

