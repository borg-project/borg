#cython: profile=False
"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import borg

cimport cython
cimport cython.parallel
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

def mean_posterior(sampler, solver_names, training, bins):
    """Sample, and return a model."""

    # XXX in the process of being dismantled

    outcomes_NSC = training.to_bins_array(solver_names, bins)
    (components_NSC,) = sampler.sample(outcomes_NSC, stored = 1)
    log_survival_NSC = borg.statistics.to_log_survival(components_NSC, axis = -1)
    log_masses_NSC = borg.statistics.floored_log(components_NSC)

    return \
        MultinomialModel(
            training.get_common_budget() / bins,
            log_survival_NSC,
            log_masses = log_masses_NSC,
            features = training.to_features_array(),
            )

class MultinomialModel(object):
    """Multinomial mixture model."""

    def __init__(
        self,
        interval,
        log_survival,
        log_weights = None,
        log_masses = None,
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
        self._features = features

        borg.statistics.assert_log_weights(self._log_weights_N)
        borg.statistics.assert_log_survival(self._log_survival_NSC, 2)
        borg.statistics.assert_log_probabilities(self._log_masses_NSC)

    def with_weights(self, new_log_weights):
        """Return an equivalent model with new weights."""

        return \
            MultinomialModel(
                self._interval,
                self._log_survival_NSC,
                log_weights = new_log_weights,
                log_masses = self._log_masses_NSC,
                features = self._features,
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
    def features(self):
        """Features of associates instances."""

        return self._features

class MulSampler(object):
    def __init__(self, alpha = 1e-8):
        self._alpha = alpha

    @cython.infer_types(True)
    def sample(self, counts, stored = 4, burn_in = 0, spacing = 1):
        """Sample parameters of the multinomial model using Gibbs."""

        cdef int N = counts.shape[0]
        cdef int S = counts.shape[1]
        cdef int D = counts.shape[2]

        cdef numpy.ndarray[double, ndim = 3] thetas_NSD = numpy.empty((N, S, D), numpy.double)
        cdef numpy.ndarray[double, ndim = 1] alpha_D = numpy.ones(D, numpy.double) * self._alpha
        cdef numpy.ndarray[int, ndim = 3] counts_NSD = numpy.asarray(counts, numpy.intc)

        cdef unsigned int thetas_NSD_stride2 = thetas_NSD.strides[2]
        cdef unsigned int alpha_D_stride0 = alpha_D.strides[0]
        cdef unsigned int counts_NSD_stride2 = counts_NSD.strides[2]

        samples = []

        for i in xrange(burn_in + (stored - 1) * spacing + 1):
            # sample multinomial components
            for s in xrange(S):
                for n in xrange(N):
                    borg.statistics.post_dirichlet_rv(
                        D,
                        &thetas_NSD[n, s, 0], thetas_NSD_stride2,
                        &alpha_D[0], alpha_D_stride0,
                        &counts_NSD[n, s, 0], counts_NSD_stride2,
                        )

                    borg.statistics.assert_weights(thetas_NSD[n, s, :], axis = -1)

            # record sample?
            if burn_in <= i and (i - burn_in) % spacing == 0:
                samples.append(thetas_NSD.copy())

                logger.info("recorded sample at Gibbs iteration %i", i)

        return samples

class MulEstimator(object):
    def __init__(self, alpha = 1e-8):
        self._alpha = alpha

    def __call__(self, run_data, bins, full_data):
        """Estimator parameters of the simple multinomial model."""

        counts_NSD = run_data.to_bins_array(run_data.solver_names, bins)
        samples_NSD = counts_NSD + self._alpha
        samples_NSD /= numpy.sum(samples_NSD, axis = -1)[..., None]
        features_NF = run_data.to_features_array()
        interval = run_data.get_common_budget() / bins

        return \
            MultinomialModel(
                interval,
                borg.statistics.to_log_survival(samples_NSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_NSD),
                features = features_NF,
                )

class MulDirSampler(object):
    @cython.infer_types(True)
    def sample(self, counts, stored = 4, burn_in = 1000, spacing = 100):
        """Sample parameters of the DCM model using Gibbs."""

        cdef int N = counts.shape[0]
        cdef int S = counts.shape[1]
        cdef int D = counts.shape[2]

        cdef numpy.ndarray[double, ndim = 3] thetas_NSD = numpy.empty((N, S, D), numpy.double)
        cdef numpy.ndarray[double, ndim = 2] alpha_SD = numpy.ones((S, D), numpy.double)
        cdef numpy.ndarray[int, ndim = 3] counts_NSD = numpy.asarray(counts, numpy.intc)

        cdef unsigned int thetas_NSD_stride2 = thetas_NSD.strides[2]
        cdef unsigned int alpha_SD_stride1 = alpha_SD.strides[1]
        cdef unsigned int counts_NSD_stride2 = counts_NSD.strides[2]

        theta_samples = []

        for i in xrange(burn_in + (stored - 1) * spacing + 1):
            # sample multinomial components
            for s in xrange(S):
                for n in xrange(N):
                    borg.statistics.post_dirichlet_rv(
                        D,
                        &thetas_NSD[n, s, 0], thetas_NSD_stride2,
                        &alpha_SD[s, 0], alpha_SD_stride1,
                        &counts_NSD[n, s, 0], counts_NSD_stride2,
                        )

            thetas_NSD[thetas_NSD < 1e-32] = 1e-32

            # optimize the Dirichlets
            for s in xrange(S):
                alpha_SD[s, :] = borg.statistics.dirichlet_estimate_ml(thetas_NSD[:, s, :])

            if burn_in <= i and (i - burn_in) % spacing == 0:
                theta_samples.append(thetas_NSD.copy())

                logger.info("recorded sample %i at Gibbs iteration %i", len(theta_samples), i)

            assert numpy.all(alpha_SD > 1e-32)
            assert numpy.all(numpy.isfinite(thetas_NSD))

        return theta_samples

class MulDirMixSampler(object):
    def __init__(self, K = 32):
        self._K = K

    def sample(self, counts, int stored = 1, int burn_in = 1000, int spacing = 50):
        return mul_dir_mix_fit(counts, self._K)

def mul_dir_mix_fit(counts, int K):
    counts_NSD = counts
    (N, S, D) = counts_NSD.shape
    T = N

    samples_TSD = numpy.empty((T, S, D), numpy.double)
    alphas_KSD = numpy.empty((K, S, D), numpy.double)
    log_responsibilities_KSN = numpy.empty((K, S, N), numpy.double)

    for s in xrange(S):
        print ">>>> ESTIMATING RTDS FOR SOLVER", s

        (
            alphas_KSD[:, s, :],
            log_responsibilities_KSN[:, s, :],
            ) = \
            borg.statistics.dcm_mixture_estimate_ml(counts_NSD[:, s, :], K)

    for t in xrange(T):
        n = t % N

        for s in xrange(S):
            #k = borg.statistics.categorical_rv_log(log_responsibilities_KSN[:, s, n])
            #k = numpy.argmax(log_responsibilities_KSN[:, s, n])

            #samples_TSD[t, s, :] = alphas_KSD[k, s, :] + counts_NSD[n, s, :]
            #samples_TSD[t, s, :] /= numpy.sum(samples_TSD[t, s, :])

            thetas = alphas_KSD[:, s, :] + counts_NSD[n, s, :]
            thetas /= numpy.sum(thetas, axis = -1)[..., None]
            thetas *= numpy.exp(log_responsibilities_KSN[:, s, n])[..., None]

            samples_TSD[t, s, :] = numpy.sum(thetas, axis = 0)

        #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
            #print "# instance", t
            #print counts_NSD[n]
            #print samples_TSD[t]

    assert numpy.all(samples_TSD >= 0.0)

    return [samples_TSD]

class MulDirMatMixEstimator(object):
    def __init__(self, K = 128):
        self._K = K

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
            borg.statistics.dcm_matrix_mixture_estimate_ml(counts_NSD, K)

        # extract RTD samples
        T = N * K
        #T = N
        samples_TSD = numpy.empty((T, S, D), numpy.double)
        log_weights_T = numpy.empty(T, numpy.double)
        features_TF = numpy.empty((T, F), numpy.double)

        for n in xrange(N):
            #t = n
            #k = borg.statistics.categorical_rv_log(log_responsibilities_KN[:, n])
            #k = numpy.argmax(log_responsibilities_KN[:, n])

            for k in xrange(K):
                t = n * K + k

                log_weights_T[t] = log_responsibilities_KN[k, n] - numpy.log(N)
                features_TF[t, :] = features_NF[n, :]

                for s in xrange(S):
                    samples_TSD[t, s, :] = alphas_KSD[k, s, :] + counts_NSD[n, s, :]
                    samples_TSD[t, s, :] /= numpy.sum(samples_TSD[t, s, :])

                    #thetas = alphas_KSD[:, s, :] + counts_NSD[n, s, :]
                    #thetas *= numpy.exp(log_responsibilities_KN[:, n])[..., None]

                    #samples_NSD[n, s, :] = numpy.sum(thetas, axis = 0)
                    #samples_NSD[n, s, :] /= numpy.sum(samples_NSD[n, s, :])

                    borg.statistics.assert_weights(samples_TSD[t, s, :], axis = -1)

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
    def __init__(self, int K = 16):
        self._K = K

    def __call__(self, run_data, bins):
        """Fit parameters of the log-normal linked mixture model."""

        # ...
        counts_NSD = run_data.to_bins_array(run_data.solver_names, bins)
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

        #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
            #print "&&&&"
            #print ps_KSD

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
                print samples_NSD[n]

        raise SystemExit()

        return \
            MultinomialModel(
                interval,
                borg.statistics.to_log_survival(samples_NSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_NSD),
                )

