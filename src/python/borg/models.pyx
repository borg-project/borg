#cython: profile=False
"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import borg

cimport cython
cimport libc.math
cimport numpy
cimport borg.statistics

logger = borg.get_logger(__name__, default_level = "DEBUG")

cdef extern from "math.h":
    double INFINITY

@cython.infer_types(True)
def sampled_pmfs_log_pmf(pmfs, counts):
    """Compute the log probabilities of instance runs given discrete log PMFs."""

    borg.statistics.assert_log_weights(pmfs, axis = -1)

    cdef int N = pmfs.shape[0]
    cdef int S = pmfs.shape[1]
    cdef int C = pmfs.shape[2]
    cdef int M = counts.shape[0]

    cdef numpy.ndarray[double, ndim = 3] pmfs_NSC = pmfs
    cdef numpy.ndarray[int, ndim = 3] counts_MSC = counts
    cdef numpy.ndarray[double, ndim = 2] logs_NM = numpy.empty((N, M), numpy.double)

    cdef int counts_msc

    for n in xrange(N):
        for m in xrange(M):
            logs_NM[n, m] = 0.0

            for s in xrange(S):
                sum_counts_ms = 0

                for c in xrange(C):
                    counts_msc = counts_MSC[m, s, c]

                    if counts_msc > 0:
                        sum_counts_ms += counts_msc

                        logs_NM[n, m] += counts_msc * pmfs_NSC[n, s, c]
                        logs_NM[n, m] -= libc.math.lgamma(1.0 + counts_msc)

                logs_NM[n, m] += libc.math.lgamma(1.0 + sum_counts_ms)

    borg.statistics.assert_log_probabilities(logs_NM)

    return logs_NM

def run_data_log_probabilities(model, testing, weights = None):
    """Compute per-instance log probabilities of run data under a model."""

    logger.info("scoring model on %i instances", len(testing))

    (N, S, C) = model.log_masses.shape
    B = C - 1

    counts = testing.to_bins_array(testing.solver_names, B)
    log_probabilities = borg.models.sampled_pmfs_log_pmf(model.log_masses, counts)

    if weights is None:
        weights = numpy.ones_like(log_probabilities.T) / log_probabilities.shape[0]

    borg.statistics.assert_weights(weights, axis = -1)

    assert weights.T.shape == log_probabilities.shape

    weighted_lps = log_probabilities + numpy.log(weights.T)
    lps_per = numpy.logaddexp.reduce(weighted_lps, axis = 0)

    return lps_per

class MultinomialModel(object):
    """Multinomial mixture model."""

    def __init__(
        self,
        interval,
        log_survival,
        log_weights = None,
        log_masses = None,
        names = None,
        features = None,
        ):
        """Initialize."""

        (N, _, _) = log_survival.shape

        self._interval = interval
        self._log_survival_NSC = log_survival

        if log_weights is None:
            self._log_weights_N = numpy.zeros(N) - numpy.log(N)
        else:
            self._log_weights_N = log_weights

        self._log_masses_NSC = log_masses
        self._names = names
        self._features = features

        borg.statistics.assert_log_weights(self._log_weights_N)
        borg.statistics.assert_log_survival(self._log_survival_NSC, 2)
        borg.statistics.assert_log_probabilities(self._log_masses_NSC)

    def with_weights(self, new_log_weights):
        """Return an equivalent model with new weights."""

        return self.with_new(log_weights = new_log_weights)

    def with_new(self, log_weights = None, features = None):
        """Return an equivalent model with new weights."""

        log_weights = self._log_weights_N if log_weights is None else log_weights
        features = self._features if features is None else features

        return \
            MultinomialModel(
                self._interval,
                self._log_survival_NSC,
                log_weights = log_weights,
                log_masses = self._log_masses_NSC,
                names = self._names,
                features = features,
                )

    def condition(self, failures):
        """Return a model conditioned on past runs."""

        log_post_weights_N = numpy.copy(self._log_weights_N)

        for (s, b) in failures:
            log_post_weights_N += self._log_survival_NSC[:, s, b]

        log_post_weights_N -= numpy.logaddexp.reduce(log_post_weights_N)

        return MultinomialModel(self._interval, self._log_survival_NSC, log_post_weights_N)

    @property
    def interval(self):
        """The associated discretization interval."""

        return self._interval

    @property
    def log_weights(self):
        """Log weights of the model components."""

        return self._log_weights_N

    @property
    def log_survival(self):
        """Possible log discrete survival functions."""

        return self._log_survival_NSC

    @property
    def log_masses(self):
        """Possible log discrete mass functions."""

        return self._log_masses_NSC

    @property
    def names(self):
        """Names of associated instances."""

        return self._names

    @property
    def features(self):
        """Features of associated instances."""

        return self._features

class MulEstimator(object):
    def __init__(self, alpha = 1e-2):
        self._alpha = alpha

    def __call__(self, run_data, bins, full_data):
        """Estimator parameters of the simple multinomial model."""

        counts_NSD = run_data.to_bins_array(run_data.solver_names, bins)
        samples_NSD = counts_NSD + self._alpha

        # XXX hack
        #samples_NSD[..., -1] += numpy.mean(counts_NSD[..., :-1] * numpy.arange(bins), axis = -1) * 1.0

        samples_NSD /= numpy.sum(samples_NSD, axis = -1)[..., None]

        return \
            MultinomialModel(
                run_data.get_common_budget() / bins,
                borg.statistics.to_log_survival(samples_NSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_NSD),
                names = numpy.array(sorted(run_data.ids)),
                features = run_data.to_features_array(),
                )

class MulDirEstimator(object):
    def __call__(self, run_data, bins, full_data):
        counts_NSD = run_data.to_bins_array(run_data.solver_names, bins)

        (N, S, D) = counts_NSD.shape

        alphas_SD = numpy.empty((S, D), numpy.double)

        for s in xrange(S):
            alphas_SD[s, :] = borg.statistics.dcm_estimate_ml_wallach(counts_NSD[:, s, :])

        samples_NSD = counts_NSD + alphas_SD
        samples_NSD /= numpy.sum(samples_NSD, axis = -1)[..., None]

        return \
            MultinomialModel(
                run_data.get_common_budget() / bins,
                borg.statistics.to_log_survival(samples_NSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_NSD),
                names = numpy.array(sorted(run_data.ids)),
                features = run_data.to_features_array(),
                )

class MulDirMixEstimator(object):
    def __init__(self, K = 4, alpha = None, samples_per = 64):
        self._K = K
        self._alpha = alpha
        self._samples_per = samples_per

    def __call__(self, run_data, bins, full_data):
        # ...
        counts_NSD = run_data.to_bins_array(run_data.solver_names, bins)
        features_NF = run_data.to_features_array()
        interval = run_data.get_common_budget() / bins

        (N, S, D) = counts_NSD.shape
        (_, F) = features_NF.shape
        K = self._K
        T = N * self._samples_per

        # fit model parameters
        alphas_KSD = numpy.empty((K, S, D), numpy.double)
        log_responsibilities_KSN = numpy.empty((K, S, N), numpy.double)

        for s in xrange(S):
            logger.info(">>>> ESTIMATING RTDS FOR SOLVER %i", s)

            (
                alphas_KSD[:, s, :],
                log_responsibilities_KSN[:, s, :],
                ) = \
                borg.statistics.dcm_mixture_estimate_ml(counts_NSD[:, s, :], K, self._alpha)

        # sample individual distributions
        logger.info("sampling %i RTD sets under dirmix", T)

        samples_TSD = numpy.empty((T, S, D), numpy.double)
        features_TF = numpy.empty((T, F), numpy.double)
        names_N = sorted(run_data.ids)
        names_T = numpy.empty(T, object)

        for t in xrange(T):
            n = t % N

            features_TF[t, :] = features_NF[n, :]
            names_T[t] = names_N[n]

            for s in xrange(S):
                k = borg.statistics.categorical_rv_log(log_responsibilities_KSN[:, s, n])

                #samples_TSD[t, s, :] = alphas_KSD[k, s, :] + counts_NSD[n, s, :] + 1e-1
                samples_TSD[t, s, :] = alphas_KSD[k, s, :] + counts_NSD[n, s, :] + 1e-2
                samples_TSD[t, s, :] /= numpy.sum(samples_TSD[t, s, :])

        assert numpy.all(samples_TSD >= 0.0)

        return \
            MultinomialModel(
                interval,
                borg.statistics.to_log_survival(samples_TSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_TSD),
                names = names_T,
                features = features_TF,
                )

class MulDirMatMixEstimator(object):
    def __init__(self, K = 32, alpha = None):
        self._K = K
        self._alpha = alpha

    def __call__(self, run_data, bins, full_data):
        # ...
        counts_NSD = run_data.to_bins_array(run_data.solver_names, bins)
        features_NF = run_data.to_features_array()
        full_NSD = full_data.to_bins_array(full_data.solver_names, bins)
        interval = run_data.get_common_budget() / bins

        (N, S, D) = counts_NSD.shape
        (_, F) = features_NF.shape
        K = self._K

        # fit model
        (alphas_KSD, log_responsibilities_KN) = \
            borg.statistics.dcm_matrix_mixture_estimate_ml(counts_NSD, K, self._alpha)

        # extract RTD samples
        T = N * K
        #T = N
        #T = K
        samples_TSD = numpy.empty((T, S, D), numpy.double)
        log_weights_T = numpy.empty(T, numpy.double)
        features_TF = numpy.empty((T, F), numpy.double)
        names_N = sorted(run_data.ids)
        names_T = numpy.empty(T, object)

        #samples_TSD = alphas_KSD / numpy.sum(alphas_KSD, axis = -1)[..., None] # XXX
        #log_weights_T = numpy.logaddexp.reduce(log_responsibilities_KN, axis = -1) - numpy.log(N)

        #for t in xrange(T):
            #for s in xrange(S):
                #z = numpy.argmax(samples_TSD[t, s])
                #samples_TSD[t, s, :] = 0
                #samples_TSD[t, s, z] = 1

            ##with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                ##print
                ##print "@ component", t, "; weight", numpy.exp(log_weights_T[t]) * N
                ##print repr(alphas_KSD[t])
                ##print samples_TSD[t]
                ##print "----"
                ##print counts_NSD[numpy.argsort(log_responsibilities_KN[t, :])[-4:][::-1]]

        #return \
            #MultinomialModel(
                #interval,
                #borg.statistics.to_log_survival(samples_TSD, axis = -1),
                #log_masses = borg.statistics.floored_log(samples_TSD),
                #log_weights = log_weights_T,
                ##features = features_TF,
                #)

        for n in xrange(N):
            #t = n
            #k = borg.statistics.categorical_rv_log(log_responsibilities_KN[:, n])
            #k = numpy.argmax(log_responsibilities_KN[:, n])

            #log_weights_T[t] = -numpy.log(T)

            #map_KSD = alphas_KSD + counts_NSD[None, n]
            #map_KSD /= numpy.sum(map_KSD, axis = -1)[..., None]
            #map_KSD *= numpy.exp(log_responsibilities_KN[:, n])[..., None, None]
            #map_SD = numpy.sum(map_KSD, axis = 0)

            ##map_SD = alphas_KSD[k]
            ##map_SD /= numpy.sum(map_SD, axis = -1)[..., None]

            #for s in xrange(S):
                ##samples_TSD[t, s, :] = 0.0
                ##samples_TSD[t, s, numpy.argmax(map_SD[s, :])] = 1.0
                #samples_TSD[t, s, :] = map_SD[s, :]

                #borg.statistics.assert_weights(samples_TSD[t, s, :], axis = -1)

            #features_TF[t, :] = features_NF[n, :]

            #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                #print "# t =", t, "n =", n
                #print counts_NSD[n]
                #print full_NSD[sorted(full_data.ids).index(sorted(run_data.ids)[n])]
                #print samples_TSD[t]

            for k in xrange(K):
                t = n * K + k

                log_weights_T[t] = log_responsibilities_KN[k, n] - numpy.log(N)
                features_TF[t, :] = features_NF[n, :]
                names_T[t] = names_N[n]

                for s in xrange(S):
                    samples_TSD[t, s, :] = alphas_KSD[k, s, :] + counts_NSD[n, s, :] + 1e-2
                    samples_TSD[t, s, :] /= numpy.sum(samples_TSD[t, s, :])
                    #samples_TSD[t, s, :] = 0.0
                    #samples_TSD[t, s, numpy.argmax(alphas_KSD[k, s, :])] = 1.0

                    #thetas = alphas_KSD[:, s, :] + counts_NSD[n, s, :]
                    #thetas *= numpy.exp(log_responsibilities_KN[:, n])[..., None]

                    #samples_NSD[n, s, :] = numpy.sum(thetas, axis = 0)
                    #samples_NSD[n, s, :] /= numpy.sum(samples_NSD[n, s, :])

                    borg.statistics.assert_weights(samples_TSD[t, s, :], axis = -1)

                #if numpy.exp(log_responsibilities_KN[k, n]) > 1e-1:
                    #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                        #print "# t =", t, "n =", n, "k =", k, "({0})".format(numpy.exp(log_responsibilities_KN[k, n]))
                        #print counts_NSD[n]
                        #print full_NSD[sorted(full_data.ids).index(sorted(run_data.ids)[n])]
                        #print alphas_KSD[k]
                        #print samples_TSD[t]

        #raise SystemExit()

        assert numpy.all(samples_TSD >= 0.0)

        return \
            MultinomialModel(
                interval,
                borg.statistics.to_log_survival(samples_TSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_TSD),
                log_weights = log_weights_T,
                names = names_T,
                features = features_TF,
                )

class DiscreteLogNormalMixEstimator(object):
    def __init__(self, int K = 32):
        self._K = K

    def __call__(self, run_data, bins):
        """Fit parameters of the log-normal mixture model."""

        # ...
        counts_NSD = run_data.to_bins_array(run_data.solver_names, bins)
        budget = run_data.get_common_budget()
        interval = budget / bins

        (N, S, D) = counts_NSD.shape
        K = self._K

        # estimate training RTDs
        ps_SKD = numpy.empty((S, K, D), numpy.double)
        log_responsibilities_SKN = numpy.empty((S, K, N), numpy.double)

        for s in xrange(S):
            print ">>>> ESTIMATING RTDS FOR SOLVER", s

            (
                ps_SKD[s, :, :],
                log_responsibilities_SKN[s, :, :],
                ) = \
                borg.statistics.discrete_log_normal_mixture_estimate_ml(
                    counts_NSD[:, s, :],
                    budget,
                    K,
                    )

            borg.statistics.assert_log_weights(log_responsibilities_SKN[s, :], axis = 0)

        samples_NSD = numpy.empty((N, S, D))

        for n in xrange(N):
            for s in xrange(S):
                #k = borg.statistics.categorical_rv_log(log_post_weights_KSN[:, s, n])
                #k = numpy.argmax(log_post_weights_KSN[:, s, n])

                thetas = numpy.copy(ps_SKD[s, :, :])
                thetas *= numpy.exp(log_responsibilities_SKN[s, :, n])[..., None]

                samples_NSD[n, s, :] = numpy.sum(thetas, axis = 0)

            with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                print "# instance", n
                print counts_NSD[n]
                print samples_NSD[n]

        #raise SystemExit()

        return \
            MultinomialModel(
                interval,
                borg.statistics.to_log_survival(samples_NSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_NSD),
                )

class DiscreteLogNormalMatMixEstimator(object):
    def __init__(self, int K = 128):
        self._K = K

    def __call__(self, run_data, bins, full_data):
        """Fit parameters of the log-normal linked mixture model."""

        # ...
        counts_NSD = run_data.to_bins_array(run_data.solver_names, bins)
        full_NSD = full_data.to_bins_array(full_data.solver_names, bins)
        budget = run_data.get_common_budget()
        interval = budget / bins

        (N, S, D) = counts_NSD.shape
        K = self._K

        # estimate training RTDs
        (ps_KSD, log_responsibilities_KN) = \
            borg.statistics.discrete_log_normal_matrix_mixture_estimate_ml(
                counts_NSD,
                budget,
                K,
                )

        borg.statistics.assert_log_weights(log_responsibilities_KN, axis = 0)

        with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
            print "&&&&"
            print ps_KSD

        samples_TSD = ps_KSD
        log_weights_T = numpy.logaddexp.reduce(log_responsibilities_KN, axis = -1) - numpy.log(N)

        #for t in xrange(T):
            ##for s in xrange(S):
                ##z = numpy.argmax(samples_TSD[t, s])
                ##samples_TSD[t, s, :] = 0
                ##samples_TSD[t, s, z] = 1

            #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                #print
                #print "@ component", t, "; weight", numpy.exp(log_weights_T[t]) * N
                #print repr(alphas_KSD[t])
                #print samples_TSD[t]
                #print "----"
                #print counts_NSD[numpy.argsort(log_responsibilities_KN[t, :])[-4:][::-1]]

        return \
            MultinomialModel(
                interval,
                borg.statistics.to_log_survival(samples_TSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_TSD),
                log_weights = log_weights_T,
                #features = features_TF,
                )

        samples_NSD = numpy.empty((N, S, D))

        for n in xrange(N):
            for s in xrange(S):
                #k = borg.statistics.categorical_rv_log(log_post_weights_KSN[:, s, n])
                #k = numpy.argmax(log_post_weights_KSN[:, s, n])

                thetas = numpy.copy(ps_KSD[:, s, :])
                thetas *= numpy.exp(log_responsibilities_KN[:, n])[..., None]

                samples_NSD[n, s, :] = numpy.sum(thetas, axis = 0)

            with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                print "# instance", n
                print counts_NSD[n]
                print full_NSD[sorted(full_data.ids).index(sorted(run_data.ids)[n])]
                print samples_NSD[n]

        #raise SystemExit()

        return \
            MultinomialModel(
                interval,
                borg.statistics.to_log_survival(samples_NSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_NSD),
                )

