"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import scipy.stats
import sklearn.linear_model
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

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

    def __init__(self, run_data, model):
        """Initialize."""

        (features_NF, ps_NM) = prepare_training_data_raw(run_data, model)

        self._regression = sklearn.linear_model.Ridge()

        self._regression.fit(features_NF, ps_NM)

    def predict(self, task, features, normalize = True):
        """Predict RTD probabilities."""

        features = numpy.asarray(features)

        prediction = self._regression.predict(features)
        prediction = numpy.clip(prediction / 100.0, 0.0, 1.0)
        #prediction = 1.0 / (1.0 + numpy.exp(-prediction))

        if normalize:
            prediction /= numpy.sum(prediction)

        return prediction

class ClusterRegression(object):
    """Cluster regression model."""

    def __init__(self, run_data, model):
        """Initialize."""

        self._run_data = run_data
        self._model = model
        self._feature_values_NF = run_data.to_features_array()

        self._feature_mus_F = numpy.mean(self._feature_values_NF, axis = 0)
        self._feature_sigmas_F = numpy.std(self._feature_values_NF, axis = 0)
        self._feature_sigmas_F += 1e-4
        #self._feature_mus_F = numpy.zeros(self._feature_values_NF.shape[1])
        #self._feature_sigmas_F = numpy.ones(self._feature_values_NF.shape[1])

        self._feature_values_NF -= self._feature_mus_F
        self._feature_values_NF /= self._feature_sigmas_F

    def predict(self, instance, features, normalize = True):
        """Predict RTD probabilities."""

        (N, F) = self._feature_values_NF.shape

        all_features_NF = self._feature_values_NF
        instance_features_F = (features - self._feature_mus_F) / self._feature_sigmas_F
        log_t_pdfs_NF = scipy.stats.t.logpdf(instance_features_F[None, ...], 1e-2, loc = all_features_NF)
        log_weights_N = -numpy.ones(N) * N
        log_weights_N += numpy.sum(log_t_pdfs_NF, axis = -1)
        log_weights_N -= numpy.logaddexp.reduce(log_weights_N)

        n = sorted(self._run_data.run_lists).index(instance)
        print log_weights_N[n:n + 8]

        return numpy.exp(log_weights_N)

