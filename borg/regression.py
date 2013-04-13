"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import sklearn.svm
import sklearn.pipeline
import sklearn.linear_model
import sklearn.decomposition
import sklearn.kernel_approximation
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

class MultiClassifier(object):
    def __init__(self, model_class, **model_kwargs):
        self._model_class = model_class
        self._model_kwargs = model_kwargs

    def fit(self, X, Y):
        (N, D) = Y.shape
        (_, self._F) = X.shape

        logger.info("fitting %i models to %i examples", D, N)

        self._models = [None] * D

        for d in xrange(D):
            if d % 250 == 0:
                logger.info("fit %i models so far", d)

            if numpy.any(Y[:, d] > 0):
                model = self._model_class(**self._model_kwargs)

                model.fit(X, Y[:, d], class_weight = {0: 1.0, 1: 10.0})
            else:
                model = None

            self._models[d] = model

        return self

    def predict_log_proba(self, X):
        (M, F) = X.shape
        D = len(self._models)
        z = numpy.empty((M, D))

        for (d, model) in enumerate(self._models):
            if model is None:
                z[:, d] = 0.0
            else:
                # TODO use predict_log_proba when it stops tossing warnings
                z[:, d] = numpy.log(model.predict_proba(X)[:, 1] + 1e-64)

        return z

    def get_feature_weights(self):
        coefs_list = [] 

        for model in self._models:
            if model is None:
                coefs_list.append([0.0] * self._F)
            else:
                assert model.coef_.shape == (1, self._F)

                coefs_list.append(model.coef_[0])

        coefs = numpy.array(coefs_list)
        weights = numpy.mean(numpy.abs(coefs), axis = 0)

        return weights

def mapify_model_survivals(model):
    """Compute per-instance MAP survival functions."""

    (P, S, D) = model.log_masses.shape
    (_, F) = model.features.shape
    unique_names = numpy.unique(model.names)
    (N,) = unique_names.shape
    masks = numpy.empty((N, P), bool)
    map_survivals = numpy.empty((N, S, D))
    features = numpy.empty((N, F))

    logger.info("computing MAP RTDs over %i samples", P)

    for (n, name) in enumerate(unique_names):
        masks[n] = mask = model.names == name
        features[n] = model.features[mask][0]

        log_survivals = model.log_survival[mask]
        log_weights = model.log_weights[mask]
        log_weights -= numpy.logaddexp.reduce(log_weights)

        map_survivals[n, :, :] = numpy.logaddexp.reduce(log_survivals + log_weights[:, None, None])

    return (unique_names, masks, features, numpy.exp(map_survivals))

class NearestRTDRegression(object):
    """Predict nearest RTDs."""

    def __init__(self, model):
        self._model = model
        (names, self._masks, features, survivals) = mapify_model_survivals(model)
        (N, _) = features.shape

        logger.info("computing %i^2 == %i inter-RTD distances", N, N * N)

        distances = borg.bregman.survival_distances_all(survivals)
        nearest = numpy.zeros((N, N), dtype = numpy.intc)
        nearest_count = min(32, N / 4)

        for n in xrange(N):
            nearest[n, numpy.argsort(distances[n])[:nearest_count]] = 1

        logger.info("fitting classifier to nearest RTDs")

        classifier = MultiClassifier(sklearn.linear_model.LogisticRegression)
        #classifier = MultiClassifier(sklearn.svm.SVC, scale_C = True, probability = True)
        #classifier = MultiClassifier(sklearn.linear_model.LogisticRegression, penalty = "l1", C = 1e-1)
        #classifier = MultiClassifier(sklearn.linear_model.LogisticRegression, penalty = "l2", C = 1e-2)

        self._regression = \
            sklearn.pipeline.Pipeline([
                #("pca", sklearn.decomposition.PCA(whiten = True)),
                #("kernel", sklearn.kernel_approximation.RBFSampler(n_components = 1000)),
                ("scaler", sklearn.preprocessing.Scaler()),
                ("classifier", classifier),
                ]) \
                .fit(features, nearest)

    def predict(self, tasks, features):
        """Predict RTD probabilities."""

        features = numpy.asarray(features)

        (P,) = self._model.log_weights.shape
        (N, _) = self._masks.shape
        (M, F) = features.shape

        predictions = self._regression.predict_log_proba(features)
        weights = numpy.empty((M, P))

        weights[:, :] = self._model.log_weights[None, :]

        for n in xrange(N):
            weights[:, self._masks[n]] += predictions[:, n, None]

        weights = numpy.exp(weights)
        weights += 1e-64
        weights /= numpy.sum(weights, axis = -1)[..., None]

        return weights

    @property
    def classifier(self):
        (_, classifier) = self._regression.steps[-1]

        return classifier

