"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import sklearn.svm
import sklearn.pls
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

    def __init__(self, model, cluster = cluster_kl, K = 16):
        """Initialize."""

        logger.info("clustering %i RTD samples", model.log_masses.shape[0])

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
                #("estimator", sklearn.linear_model.LogisticRegression(C = 1e-1)),
                ("estimator", sklearn.linear_model.LogisticRegression(C = 3.0)),
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

                weights[m, n] = prediction[m, self._indices[k]] / self._sizes[k] + 1e-64

        weights /= numpy.sum(weights, axis = -1)[..., None]

        return weights

class NeighborsRegression(object):
    """Nearest-neighbor model."""

    def __init__(self, model, K = 16):
        """Initialize."""

        logger.info("fitting ball tree to instances")

        features = model.features

        self._model = model
        self._K = min(K, model.log_masses.shape[0] / 2)
        self._scaler = sklearn.preprocessing.Scaler().fit(features)
        self._ball_tree = sklearn.neighbors.BallTree(self._scaler.transform(features))

    def predict(self, tasks, features):
        """Predict RTD probabilities."""

        features = self._scaler.transform(features)

        (M, F) = features.shape
        (N, _, _) = self._model.log_masses.shape

        neighbors = self._ball_tree.query(features, k = self._K, return_distance = False)
        weights = numpy.zeros((M, N)) + 1e-64

        for m in xrange(M):
            weights[m, neighbors[m]] = 1.0 / neighbors.shape[1]

        weights /= numpy.sum(weights, axis = -1)[..., None]

        return weights

class MultiSVR(object):
    def __init__(self, **svr_kwargs):
        self._svr_kwargs = svr_kwargs

    def fit(self, X, Y):
        (N, D) = Y.shape

        logger.info("fitting %i SVR models to %i examples", D, N)

        self._models = [sklearn.svm.SVR(**self._svr_kwargs) for _ in xrange(D)]

        for (d, model) in enumerate(self._models):
            print d
            model.fit(X, Y[:, d])

        return self

    def predict(self, X):
        (M, F) = X.shape
        D = len(self._models)
        z = numpy.empty((M, D))

        for (d, model) in enumerate(self._models):
            z[:, d] = model.predict(X)

        return z

class RTDWeightsRegression(object):
    """Predict inter-RTD distances."""

    def __init__(self, model, test_data):
        (N, S, D) = model.log_masses.shape

        logger.info("computing %i inter-RTD distances", N * N)

        survivals = numpy.exp(model.log_survival)
        distances = borg.bregman.survival_distances_all(survivals)
        sigma = 1e-1
        weights = numpy.exp(-distances**2.0 / (2.0 * sigma**2.0))

        with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
            print weights

        logger.info("fitting discriminative model to weights")

        features_NF = model.features

        assert features_NF.shape[0] == model.log_masses.shape[0]

        import sklearn.pls
        import sklearn.neighbors
        self._model = model
        self._test_model = borg.models.MulEstimator()(test_data, D - 1, test_data)
        self._regression = \
            sklearn.pipeline.Pipeline([
                ("scaler", sklearn.preprocessing.Scaler()),
                #("estimator", sklearn.linear_model.Ridge(alpha = 1e-1)),
                ("estimator", MultiSVR(kernel = "rbf", C = 1.0, gamma = 0.001)),
                #("estimator", sklearn.neighbors.KNeighborsRegressor(n_neighbors = 16)),
                #("estimator", sklearn.pls.PLSRegression()),
                ]) \
                .fit(features_NF, weights)

        #import pickle
        #with open("features.pickle", "wb") as out_file:
            #pickle.dump(features_NF, out_file)
        #with open("weights.pickle", "wb") as out_file:
            #pickle.dump(weights, out_file)

        (_, estimator) = self._regression.steps[-1]

    def predict(self, tasks, features):
        """Predict RTD probabilities."""

        features = numpy.asarray(features)

        (N,) = self._model.log_weights.shape
        (M, F) = features.shape

        prediction = self._regression.predict(features)
        prediction = numpy.clip(prediction, 0.0, 1e2)
        weights = numpy.empty((M, N))
        #sigma = 4e-1

        for m in xrange(M):
            #weights[m] = numpy.exp(-distances[m]**2.0 / (2 * sigma**2.0))
            weights[m] = prediction[m]

        weights += 1e-64
        weights /= numpy.sum(weights, axis = -1)[..., None]

        with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
            print weights * N

        return weights

    #def predict_(self, tasks, features):
        #"""Predict RTD probabilities."""

        #features = numpy.asarray(features)

        #(N,) = self._model.log_weights.shape
        #(M, F) = features.shape

        #weights = numpy.empty((M, N))
        #sigma = 0.1

        #for m in xrange(M):
            #t = list(self._test_model.names).index(tasks[m])
            #true_survival = numpy.exp(self._test_model.log_survival[t])
            #distances = borg.bregman.survival_distances(true_survival, numpy.exp(self._model.log_survival))

            #weights[m] = numpy.exp(-distances**2.0 / (2 * sigma**2.0))
            ##weights[m] = (1.0 - distances)**4.0
            ##weights[m, :] = 1e-64
            ##weights[m, numpy.argmin(distances)] = 1.0

        #weights /= numpy.sum(weights, axis = -1)[..., None]

        #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
            #print weights * N

        #return weights

class MultiClassifier(object):
    def __init__(self, model_class, **model_kwargs):
        self._model_class = model_class
        self._model_kwargs = model_kwargs

    def fit(self, X, Y):
        (N, D) = Y.shape

        logger.info("fitting %i models to %i examples", D, N)

        self._models = [self._model_class(**self._model_kwargs) for _ in xrange(D)]

        for (d, model) in enumerate(self._models):
            model.fit(X, Y[:, d], class_weight = {0: 1.0, 1: 10.0})

        return self

    def predict(self, X):
        (M, F) = X.shape
        D = len(self._models)
        z = numpy.empty((M, D))

        for (d, model) in enumerate(self._models):
            z[:, d] = model.predict(X)

        return z

    def predict_proba(self, X):
        (M, F) = X.shape
        D = len(self._models)
        z = numpy.empty((M, D))

        for (d, model) in enumerate(self._models):
            p = model.predict_proba(X)

            if p.shape[1] > 1:
                z[:, d] = p[:, 1]
            else:
                z[:, d] = 0.0

        return z

class NearestRTDRegression(object):
    """Predict nearest RTDs."""

    def __init__(self, model, test_data):
        (N, S, D) = model.log_masses.shape

        logger.info("computing %i inter-RTD distances", N * N)

        survivals = numpy.exp(model.log_survival)
        distances = borg.bregman.survival_distances_all(survivals)
        nearest = numpy.zeros((N, N), dtype = numpy.intc)

        for n in xrange(N):
            nearest[n, numpy.argsort(distances[n])[:32]] = 1

        logger.info("fitting classifier to nearest RTDs")

        features_NF = model.features

        assert features_NF.shape[0] == model.log_masses.shape[0]

        classifier = MultiClassifier(sklearn.linear_model.LogisticRegression, C = 8e-1)
        #classifier = MultiClassifier(sklearn.svm.SVC, scale_C = False, probability = True)
        self._model = model
        self._regression = \
            sklearn.pipeline.Pipeline([
                ("scaler", sklearn.preprocessing.Scaler()),
                ("classifier", classifier),
                ]) \
                .fit(features_NF, nearest)

        #import pickle
        #with open("nearest.pickle", "wb") as out_file:
            #pickle.dump(nearest, out_file)

    def predict(self, tasks, features):
        """Predict RTD probabilities."""

        features = numpy.asarray(features)

        (N,) = self._model.log_weights.shape
        (M, F) = features.shape

        weights = self._regression.predict_proba(features)
        weights += 1e-64
        weights /= numpy.sum(weights, axis = -1)[..., None]

        with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
            print weights * N

        return weights

