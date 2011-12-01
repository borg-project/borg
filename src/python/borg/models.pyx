#cython: profile=False
"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import borg
import cargo

cimport cython
cimport libc.math
cimport numpy
cimport borg.statistics

logger = cargo.get_logger(__name__, default_level = "DEBUG")

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

def run_data_log_probabilities(model, B, testing, weights = None):
    """Compute per-instance log probabilities of run data under a model."""

    logger.info("scoring model on %i instances", len(testing))

    (N, S, C) = model.log_masses.shape
    B = C - 1

    counts = testing.to_bins_array(testing.solver_names, B)
    log_probabilities = borg.models.sampled_pmfs_log_pmf(model.log_masses, counts)

    if weights is None:
        weights = numpy.ones_like(log_probabilities.T) / log_probabilities.shape[1]

    borg.statistics.assert_weights(weights, axis = -1)

    assert weights.T.shape == log_probabilities.shape

    weighted_lps = log_probabilities + numpy.log(weights.T)
    lps_per = numpy.logaddexp.reduce(weighted_lps, axis = 0)

    return lps_per

def run_data_mean_log_probability(model, testing, weights = None):
    """Compute mean log probability of run data under a model."""

    return numpy.mean(run_data_log_probabilities(model, None, testing, weights = weights))

class MultinomialModel(object):
    """Multinomial mixture model."""

    def __init__(self, interval, log_survival, log_weights = None, log_masses = None, clusters = None):
        """Initialize."""

        (N, _, _) = log_survival.shape

        self._interval = interval
        self._log_survival_NSC = log_survival

        if log_weights is None:
            self._log_weights_N = -numpy.ones(N) * numpy.log(N)
        else:
            self._log_weights_N = log_weights

        self._log_masses_NSC = log_masses
        self._clusters_NK = clusters

        borg.statistics.assert_log_weights(self._log_weights_N)
        borg.statistics.assert_log_survival(self._log_survival_NSC, 2)
        borg.statistics.assert_log_probabilities(self._log_masses_NSC)

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
    def clusters(self):
        """Possible cluster assignments."""

        return self._clusters_NK

class MulSampler(object):
    def __init__(self, alpha = 1e-4):
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

            # record sample?
            if burn_in <= i and (i - burn_in) % spacing == 0:
                samples.append(thetas_NSD.copy())

                logger.info("recorded sample at Gibbs iteration %i", i)

        return samples

    def fit(self, solver_names, training, B, T = 1):
        """Sample and return a model."""

        logger.info("fitting multinomial model")

        # run sampling
        N = len(training.run_lists)
        M = N * T
        S = len(solver_names)
        C = B + 1

        outcomes_NSC = training.to_bins_array(solver_names, B)
        samples = self.sample(outcomes_NSC, stored = T)
        components_MSC = numpy.vstack(samples)
        log_survival_MSC = borg.statistics.to_log_survival(components_MSC, axis = -1)
        log_masses_MSC = borg.statistics.floored_log(components_MSC)

        return MultinomialModel(training.get_common_budget() / B, log_survival_MSC, log_masses = log_masses_MSC)

class MulDirSampler(object):
    @cython.infer_types(True)
    def sample(self, counts, stored = 4, burn_in = 1024, spacing = 128):
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

        theta_samples_TNSD = numpy.empty((stored, N, S, D), numpy.double)

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
                theta_samples_TNSD[int((i - burn_in) / spacing), ...] = thetas_NSD

                logger.info("recorded sample at Gibbs iteration %i", i)

            assert numpy.all(alpha_SD > 1e-32)
            assert numpy.all(numpy.isfinite(thetas_NSD))

        return theta_samples_TNSD

    def fit(self, solver_names, training, B, T = 1):
        """Sample and return a model."""

        N = len(training.run_lists)
        M = N * T
        S = len(solver_names)
        C = B + 1

        logger.info("fitting dirichlet-multinomial model to %i instances", N)

        outcomes_NSC = training.to_bins_array(solver_names, B)
        components_MSC = numpy.empty((M, S, C), numpy.double)
        theta_samples_TNSC = self.sample(outcomes_NSC, stored = T)

        for t in xrange(T):
            components_MSC[t * N:(t + 1) * N, ...] = theta_samples_TNSC[t, ...]

        log_survival_MSC = borg.statistics.to_log_survival(components_MSC, axis = -1)

        return MultinomialModel(training.get_common_budget() / B, log_survival_MSC, log_masses = numpy.log(components_MSC))

class MulDirMixSampler(object):
    def __init__(self, K = 32):
        self._K = K

    @cython.infer_types(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def sample(self, counts, int stored = 4, int burn_in = 1000, int spacing = 100):
        """Sample parameters of the DCM mixture model using Gibbs."""

        cdef int N = counts.shape[0]
        cdef int S = counts.shape[1]
        cdef int D = counts.shape[2]
        cdef int K = self._K

        cdef numpy.ndarray[int, ndim = 3] counts_NSD = numpy.asarray(counts, numpy.intc)
        cdef numpy.ndarray[double, ndim = 1] eta_K = numpy.ones(K, numpy.double)
        cdef numpy.ndarray[double, ndim = 2] pis_SK = numpy.random.dirichlet(numpy.ones(K) + 1e-1, size = S)
        cdef numpy.ndarray[int, ndim = 2] zs_NS = numpy.random.randint(K, size = (N, S)).astype(numpy.intc)
        cdef numpy.ndarray[double, ndim = 3] alphas_SKD = numpy.ones((S, K, D), numpy.double) + 1e-1
        cdef numpy.ndarray[double, ndim = 3] thetas_NSD = numpy.empty((N, S, D), numpy.double)
        cdef numpy.ndarray[double, ndim = 1] log_posterior_K = numpy.empty(K, numpy.double)
        cdef numpy.ndarray[int, ndim = 1] z_counts_K = numpy.zeros(K, numpy.intc)

        cdef unsigned int eta_K_stride0 = eta_K.strides[0]
        cdef unsigned int pis_SK_stride1 = pis_SK.strides[1]
        cdef unsigned int thetas_NSD_stride2 = thetas_NSD.strides[2]
        cdef unsigned int alphas_SKD_stride2 = alphas_SKD.strides[2]
        cdef unsigned int counts_NSD_stride2 = counts_NSD.strides[2]
        cdef unsigned int log_posterior_K_stride0 = log_posterior_K.strides[0]
        cdef unsigned int z_counts_K_stride0 = z_counts_K.strides[0]

        theta_samples_TNSD = numpy.empty((stored, N, S, D), numpy.double)

        for i in xrange(burn_in + (stored - 1) * spacing + 1):
            # sample multinomial components
            for s in xrange(S):
                for n in xrange(N):
                    borg.statistics.post_dirichlet_rv(
                        D,
                        &thetas_NSD[n, s, 0], thetas_NSD_stride2,
                        &alphas_SKD[s, zs_NS[n, s], 0], alphas_SKD_stride2,
                        &counts_NSD[n, s, 0], counts_NSD_stride2,
                        )

            thetas_NSD[n, s, thetas_NSD[n, s] < 1e-32] = 1e-32

            # sample cluster assignments
            for s in xrange(S):
                for n in xrange(N):
                    total = 0.0

                    for k in xrange(K):
                        log_posterior_K[k] = \
                            libc.math.log(pis_SK[s, k]) + \
                            borg.statistics.dirichlet_log_pdf_raw(
                                D,
                                &alphas_SKD[s, k, 0], alphas_SKD_stride2,
                                &thetas_NSD[n, s, 0], thetas_NSD_stride2,
                                )

                        total = borg.statistics.log_plus(total, log_posterior_K[k])

                    for k in xrange(K):
                        log_posterior_K[k] -= total

                    zs_NS[n, s] = borg.statistics.categorical_rv_log_raw(K, &log_posterior_K[0], log_posterior_K_stride0)

            # optimize the Dirichlets
            for s in xrange(S):
                for k in xrange(K):
                    cluster_thetas_XD = thetas_NSD[numpy.nonzero(zs_NS[:, s] == k)[0], s, :]

                    if cluster_thetas_XD.size > 0:
                        alphas_SKD[s, k, :] = borg.statistics.dirichlet_estimate_ml(cluster_thetas_XD)
                    else:
                        alphas_SKD[s, k, :] = numpy.random.gamma(1.1, 1, size = D)

            alphas_SKD += 1e-32

            # sample pis
            for s in xrange(S):
                for k in xrange(K):
                    z_counts_K[k] = 0

                for n in xrange(N):
                    z_counts_K[zs_NS[n, s]] += 1

                borg.statistics.post_dirichlet_rv(
                    K,
                    &pis_SK[s, 0], pis_SK_stride1,
                    &eta_K[0], eta_K_stride0,
                    &z_counts_K[0], z_counts_K_stride0,
                    )

            if burn_in <= i and (i - burn_in) % spacing == 0:
                theta_samples_TNSD[int((i - burn_in) / spacing), ...] = thetas_NSD

                logger.info("recorded sample at Gibbs iteration %i", i)

        return theta_samples_TNSD

    def fit(self, solver_names, training, B, T = 1):
        """Sample and return a model."""

        logger.info("fitting Dirichlet mixture-multinomial model")

        N = len(training.run_lists)
        M = N * T
        S = len(solver_names)
        C = B + 1

        outcomes_NSC = training.to_bins_array(solver_names, B)
        components_MSC = numpy.empty((M, S, C), numpy.double)
        theta_samples_TNSC = self.sample(outcomes_NSC, stored = T)

        for t in xrange(T):
            components_MSC[t * N:(t + 1) * N, ...] = theta_samples_TNSC[t, ...]

        log_survival_MSC = borg.statistics.to_log_survival(components_MSC, axis = -1)

        return MultinomialModel(training.get_common_budget() / B, log_survival_MSC, log_masses = numpy.log(components_MSC))

class MulDirMatMixSampler(object):
    def __init__(self, K = 32):
        self._K = K

    @cython.infer_types(True)
    def sample(self, counts, int stored = 4, int burn_in = 1000, int spacing = 100):
        """Sample parameters of the DCM mixture model using Gibbs."""

        cdef int N = counts.shape[0]
        cdef int S = counts.shape[1]
        cdef int D = counts.shape[2]
        cdef int K = self._K

        cdef numpy.ndarray[int, ndim = 3] counts_NSD = numpy.asarray(counts, numpy.intc)
        cdef numpy.ndarray[double, ndim = 1] eta_K = numpy.ones(K, numpy.double)
        cdef numpy.ndarray[double, ndim = 1] pi_K = numpy.random.dirichlet(numpy.ones(K) + 1e-1)
        cdef numpy.ndarray[int, ndim = 1] zs_N = numpy.random.randint(K, size = N).astype(numpy.intc)
        cdef numpy.ndarray[double, ndim = 3] alphas_SKD = numpy.ones((S, K, D), numpy.double) + 1e-1
        cdef numpy.ndarray[double, ndim = 3] thetas_NSD = numpy.empty((N, S, D), numpy.double)
        cdef numpy.ndarray[double, ndim = 1] log_posterior_K = numpy.empty(K, numpy.double)
        cdef numpy.ndarray[int, ndim = 1] z_counts_K = numpy.empty(K, numpy.intc)

        cdef unsigned int thetas_NSD_stride2 = thetas_NSD.strides[2]
        cdef unsigned int alphas_SKD_stride2 = alphas_SKD.strides[2]
        cdef unsigned int counts_NSD_stride2 = counts_NSD.strides[2]
        cdef unsigned int log_posterior_K_stride0 = log_posterior_K.strides[0]

        theta_samples = []
        z_samples = []

        for i in xrange(burn_in + (stored - 1) * spacing + 1):
            # sample multinomial components
            for s in xrange(S):
                for n in xrange(N):
                    borg.statistics.post_dirichlet_rv(
                        D,
                        &thetas_NSD[n, s, 0], thetas_NSD_stride2,
                        &alphas_SKD[s, zs_N[n], 0], alphas_SKD_stride2,
                        &counts_NSD[n, s, 0], counts_NSD_stride2,
                        )

            thetas_NSD[n, s, thetas_NSD[n, s] < 1e-32] = 1e-32

            # sample cluster assignments
            for n in xrange(N):
                total = 0.0

                for k in xrange(K):
                    log_posterior_K[k] = libc.math.log(pi_K[k])

                    for s in xrange(S):
                        log_posterior_K[k] += \
                            borg.statistics.dirichlet_log_pdf_raw(
                                D,
                                &alphas_SKD[s, k, 0], alphas_SKD_stride2,
                                &thetas_NSD[n, s, 0], thetas_NSD_stride2,
                                )

                    total = borg.statistics.log_plus(total, log_posterior_K[k])

                for k in xrange(K):
                    log_posterior_K[k] -= total

                zs_N[n] = borg.statistics.categorical_rv_log_raw(K, &log_posterior_K[0], log_posterior_K_stride0)

            # optimize the Dirichlets
            for s in xrange(S):
                for k in xrange(K):
                    cluster_thetas_XD = thetas_NSD[numpy.nonzero(zs_N[:] == k)[0], s, :]

                    if cluster_thetas_XD.size > 0:
                        alphas_SKD[s, k, :] = borg.statistics.dirichlet_estimate_ml(cluster_thetas_XD)
                    else:
                        alphas_SKD[s, k, :] = numpy.random.gamma(1.1, 1, size = D)

            alphas_SKD += 1e-32

            # sample pis
            for k in xrange(K):
                z_counts_K[k] = 0

            for n in xrange(N):
                z_counts_K[zs_N[n]] += 1

            pi_K[:] = numpy.random.dirichlet(eta_K + z_counts_K)

            # record a sample?
            if burn_in <= i and (i - burn_in) % spacing == 0:
                theta_samples.append(thetas_NSD.copy())
                z_samples.append(zs_N.copy())

                logger.info("recorded sample at Gibbs iteration %i", i)

        return (theta_samples, z_samples)

    def fit(self, solver_names, training, B, T = 1):
        """Sample and return a model."""

        logger.info("fitting Dirichlet mixture model")

        outcomes_NSC = training.to_bins_array(solver_names, B)
        (theta_samples, z_samples) = self.sample(outcomes_NSC, stored = T)
        components_MSC = numpy.vstack(theta_samples)
        clusters_N = borg.statistics.indicator(numpy.hstack(z_samples), self._K)
        log_survival_MSC = borg.statistics.to_log_survival(components_MSC, axis = -1)

        return \
            MultinomialModel(
                training.get_common_budget() / B,
                log_survival_MSC,
                log_masses = numpy.log(components_MSC),
                clusters = clusters_N,
                )

