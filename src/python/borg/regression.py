"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import sklearn.cluster
import sklearn.pipeline
import sklearn.neighbors
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

def cluster_kl(model, K):
    """Cluster RTDs according to KL divergence."""

    kl_means = borg.bregman.KLMeans(k = K).fit(numpy.exp(model.log_masses))

    return kl_means._assignments

def plan_affinities(log_survival_WSB):
    """Compute plan-based distances between instances."""

    # ...
    (W, S, B) = log_survival_WSB.shape

    # compute individual plan probabilities
    ps_W = numpy.empty(W)

    for w in xrange(W):
        (_, log_fail_p) = \
            borg.planners.knapsack_plan(
                log_survival_WSB[w:w + 1, :, :-1],
                numpy.array([0.0]),
                give_log_fail = True,
                )

        ps_W[w] = 1.0 - numpy.exp(log_fail_p)

    affinity_WW = numpy.empty((W, W))

    for w in xrange(W):
        for v in xrange(w, W):
            (_, log_fail_p) = \
                borg.planners.knapsack_plan(
                    log_survival_WSB[[w, v], :, :-1],
                    numpy.zeros(2) - numpy.log(2),
                    give_log_fail = True,
                    )
            p = 1.0 - numpy.exp(log_fail_p)

            affinity_WW[w, v] = 1.0 - (max(ps_W[w], ps_W[v]) - p)
            affinity_WW[v, w] = affinity_WW[w, v]

            assert affinity_WW[v, w] > -1e-8
            assert affinity_WW[v, w] < 1.0 + 1e-8

        print w

    return affinity_WW

def cluster_plan(model, K):
    """Cluster RTDs according to plan similarity."""

    affinities = plan_affinities(model.log_survival)
    spectral = sklearn.cluster.SpectralClustering(k = K).fit(affinities)

    return spectral.labels_

class ClusteredLogisticRegression(object):
    """Logistic regression model with clustering."""

    def __init__(self, model, cluster = cluster_kl, K = 32):
        """Initialize."""

        logger.info("clustering %i RTD samples", model.log_masses.shape[0])

        self._model = model
        self._labels = cluster(model, K)
        self._indicators = borg.statistics.indicator(self._labels, K)

        #for k in xrange(K):
            #print "** cluster", k
            #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                #print numpy.exp(model.log_masses[self._labels == k])

        logger.info("fitting discriminative model to cluster assignments")

        features_NF = model.features

        assert features_NF.shape[0] == model.log_masses.shape[0]
        assert features_NF.shape[0] == self._labels.shape[0]

        self._regression = \
            sklearn.pipeline.Pipeline([
                ("scaler", sklearn.preprocessing.Scaler()),
                ("estimator", sklearn.linear_model.LogisticRegression(C = 1e-1)),
                ]) \
                .fit(features_NF, self._labels)

        (_, estimator) = self._regression.steps[-1]

        self._indices = dict(zip(estimator.label_, xrange(estimator.label_.shape[0])))
        self._sizes = numpy.sum(self._indicators, axis = 0)

        logger.info("cluster sizes: %s", self._sizes)

    def predict(self, features):
        """Predict RTD probabilities."""

        features = numpy.asarray(features)

        (M, F) = features.shape
        (N, K) = self._indicators.shape

        prediction = self._regression.predict_proba(features)
        weights = numpy.empty((M, N))

        for m in xrange(M):
            for n in xrange(N):
                k = self._labels[n]
                p_z = prediction[m, self._indices[k]]

                weights[m, n] = p_z / self._sizes[k]

        return weights

class NeighborsRegression(object):
    """Nearest-neighbor model."""

    def __init__(self, model, K = 16):
        """Initialize."""

        logger.info("fitting ball tree to instances")

        features = model.features

        self._model = model
        self._K = K
        self._scaler = sklearn.preprocessing.Scaler().fit(features)
        self._ball_tree = sklearn.neighbors.BallTree(self._scaler.transform(features))

    def predict(self, features):
        """Predict RTD probabilities."""

        features = self._scaler.transform(features)

        (M, F) = features.shape
        (N, _, _) = self._model.log_masses.shape

        neighbors = self._ball_tree.query(features, k = self._K, return_distance = False)
        weights = numpy.zeros((M, N)) + 1e-64

        print "neighbors are", neighbors

        for m in xrange(M):
            weights[m, neighbors[m]] = 1.0 / neighbors.shape[1]

        return weights

