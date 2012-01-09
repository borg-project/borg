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

def mixture_from_posterior(sampler, solver_names, training, bins, samples_per_chain = 1, chains = 1):
    """Sample, and return a model."""

    outcomes_NSC = training.to_bins_array(solver_names, bins)
    samples = []

    for i in xrange(chains):
        logger.info("sampling from %s (chain %i of %i)", type(sampler).__name__, i + 1, chains)

        chain_samples = sampler.sample(outcomes_NSC, stored = samples_per_chain)

        samples.extend(chain_samples)

    components_MSC = numpy.vstack(samples)
    log_survival_MSC = borg.statistics.to_log_survival(components_MSC, axis = -1)
    log_masses_MSC = borg.statistics.floored_log(components_MSC)

    return \
        MultinomialModel(
            training.get_common_budget() / bins,
            log_survival_MSC,
            log_masses = log_masses_MSC,
            )

def mean_posterior(sampler, solver_names, training, bins, samples_per_chain = 1, chains = 1):
    """Sample, and return a (mean) model."""

    outcomes_NSC = training.to_bins_array(solver_names, bins)
    samples = []

    for i in xrange(chains):
        logger.info("sampling from %s (chain %i of %i)", type(sampler).__name__, i + 1, chains)

        chain_samples = sampler.sample(outcomes_NSC, stored = samples_per_chain)

        samples.extend(chain_samples)

    components_MNSC = numpy.vstack([sample[None, ...] for sample in samples])
    components_NSC = numpy.mean(components_MNSC, axis = 0)
    log_survival_NSC = borg.statistics.to_log_survival(components_NSC, axis = -1)
    log_masses_NSC = borg.statistics.floored_log(components_NSC)

    return \
        MultinomialModel(
            training.get_common_budget() / bins,
            log_survival_NSC,
            log_masses = log_masses_NSC,
            )

def posterior(sampler, solver_names, training, bins):
    """Sample, and return a (mean) model."""

    outcomes_NSC = training.to_bins_array(solver_names, bins)
    (components_MSC, weights_M) = sampler.sample(outcomes_NSC)
    log_survival_MSC = borg.statistics.to_log_survival(components_MSC, axis = -1)
    log_masses_MSC = borg.statistics.floored_log(components_MSC)

    return \
        MultinomialModel(
            training.get_common_budget() / bins,
            log_survival_MSC,
            log_masses = log_masses_MSC,
            log_weights = borg.statistics.floored_log(weights_M),
            )

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

    @cython.infer_types(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def sample_(self, counts, int stored = 1, int burn_in = 1000, int spacing = 50):
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

        theta_samples = []

        for i in xrange(burn_in + (stored - 1) * spacing + 1):
            print "ITERATION", i

            # sample multinomial components
            for s in xrange(S):
                for n in xrange(N):
                    borg.statistics.post_dirichlet_rv(
                        D,
                        &thetas_NSD[n, s, 0], thetas_NSD_stride2,
                        &alphas_SKD[s, zs_NS[n, s], 0], alphas_SKD_stride2,
                        &counts_NSD[n, s, 0], counts_NSD_stride2,
                        )

                # XXX
                #print s
                #print alphas_SKD[s, zs_NS[:, s], :]
                #print counts_NSD[:, s, :]
                #thetas_NSD[:, s, :] = alphas_SKD[s, zs_NS[:, s], :] + counts_NSD[:, s, :]

            # XXX
            #thetas_NSD /= numpy.sum(thetas_NSD, axis = -1)[..., None]

            #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 240, threshold = 1000000):
                ##print zs_NS

                #n = 3
                #print "INSTANCE", n
                #print zs_NS[n, :]
                #print numpy.array([alphas_SKD[s, zs_NS[n, s], :] for s in xrange(S)])
                #print counts_NSD[n, :, :]
                #print thetas_NSD[n, :, :]

                #if i % 100 == 0:
                    #print "-" * 128
                    #for s in xrange(S):
                        #print "SOLVER", s
                        #print alphas_SKD[s, :, :]

            thetas_NSD[n, s, thetas_NSD[n, s] < 1e-32] = 1e-32

            # sample cluster assignments
            for s in xrange(S):
                for n in xrange(N):
                    # XXX also fix in matrix mixture
                    total = -INFINITY

                    for k in xrange(K):
                        log_posterior_K[k] = \
                            libc.math.log(pis_SK[s, k]) + \
                            borg.statistics.dirichlet_log_pdf_raw(
                                D,
                                &alphas_SKD[s, k, 0], alphas_SKD_stride2,
                                &thetas_NSD[n, s, 0], thetas_NSD_stride2,
                                )

                        #if i > 100:
                            #print "...", n, s
                            #print "pi:", pis_SK[s, k]
                            #print "alpha:", alphas_SKD[s, k]
                            #print "theta:", thetas_NSD[n, s]
                            #print "density:", log_posterior_K[k]

                        total = borg.statistics.log_plus(total, log_posterior_K[k])

                    for k in xrange(K):
                        log_posterior_K[k] -= total

                    #if i > 100:
                        #print ">>>>", n, s
                        #print total
                        #print numpy.exp(total)
                        #print log_posterior_K
                        #print numpy.exp(log_posterior_K)

                    #print numpy.sum(numpy.exp(log_posterior_K))

                    zs_NS[n, s] = borg.statistics.categorical_rv_log_raw(K, &log_posterior_K[0], log_posterior_K_stride0)

            # optimize the Dirichlets
            for s in xrange(S):
                for k in xrange(K):
                    cluster_thetas_XD = thetas_NSD[numpy.nonzero(zs_NS[:, s] == k)[0], s, :]

                    if cluster_thetas_XD.size > 0:
                        alphas_SKD[s, k, :] = borg.statistics.dirichlet_estimate_ml(cluster_thetas_XD)
                    else:
                        #alphas_SKD[s, k, :] = numpy.random.gamma(1.1, 1, size = D)
                        #print ">>>> reassigning cluster", k, "of solver", s
                        alphas_new = counts_NSD[numpy.random.randint(N), s, :].astype(float)
                        alphas_new += 1e-4
                        alphas_SKD[s, k, :] = alphas_new / numpy.sum(alphas_new)

                    ## XXX hackish clamping
                    #magnitude = numpy.sum(alphas_SKD[s, k, :])

                    #if magnitude > 1.0:
                        #alphas_SKD[s, k, :] /= magnitude
                        #alphas_SKD[s, k, :] *= 1.0

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
                theta_samples.append(thetas_NSD.copy())

                # XXX obviously absurd
                #sample = thetas_NSD.copy()
                #sample[sample < 0.05] = 0.0
                #sample /= numpy.sum(sample, axis = -1)[..., None]

                # XXX
                #alphas_NSD = [[alphas_SKD[s, zs_NS[n, s], :] + counts_NSD[n, s, :] for s in xrange(S)] for n in xrange(N)]
                #alphas_NSD = numpy.array(alphas_NSD)
                #sample = alphas_NSD / numpy.sum(alphas_NSD, axis = -1)[..., None]

                #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 240, threshold = 1000000):
                    #print sample

                #theta_samples.append(sample)

                logger.info("recorded sample %i at Gibbs iteration %i", len(theta_samples), i)

        return theta_samples

    def sample(self, counts, int stored = 1, int burn_in = 1000, int spacing = 50):
        return mul_dir_mix_fit(counts)

def mul_dir_mix_fit_old(counts):
    counts_NSD = counts
    (N, S, D) = counts_NSD.shape
    K = 64
    T = 2
    M = N * T

    #thetas_NSD = numpy.empty((N, S, D), numpy.double)
    thetas_MSD = numpy.empty((M, S, D), numpy.double)

    for s in xrange(S):
        print ">>>>", s

        (components_KD, log_responsibilities_KN) = \
            borg.statistics.dcm_mixture_estimate_ml(counts_NSD[:, s, :], K)

        # XXX MAP configuration
        #thetas_NKD = components_KD[None, ...] * numpy.exp(log_responsibilities_KN.T[..., None])
        #components_ND = numpy.sum(thetas_NKD, axis = 1)

        # XXX ML configuration
        #ks_N = numpy.argmax(log_responsibilities_KN, axis = 0)
        #components_ND = components_KD[ks_N, :]

        #thetas_NSD[:, s, :] = components_ND
        #thetas_NSD[:, s, :] += counts_NSD[:, s, :]
        #thetas_NSD[:, s, :] /= numpy.sum(thetas_NSD[:, s, :], axis = -1)[..., None]

        #assert numpy.all(thetas_NSD[:, s, :] >= -1e8)
        #assert numpy.min(thetas_NSD[:, s, :]) >= 0.0

        #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 240, threshold = 1000000):
            #print "counts:"
            #print counts_NSD[:, s, :]
            #print "components:"
            #print components_KD
            #print "thetas:"
            #print thetas_NSD[:, s, :]

        # XXX weighted ML configuration
        ks_TN = numpy.argsort(log_responsibilities_KN, axis = 0)[-T:, :][::-1, :]
        log_weights_TN = numpy.sort(log_responsibilities_KN, axis = 0)[-2:, :][::-1, :]
        log_weights_TN -= numpy.logaddexp.reduce(log_weights_TN, axis = 0)[None, :]
        log_weights_TN -= numpy.log(N)

        print "N =", N
        print "K =", K

        for t in xrange(T):
            with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                print "T:", t
                #print ks_TN[t, :]
                #print numpy.sort(log_responsibilities_KN, axis = 0)[:, -16:]

            thetas_ND = components_KD[ks_TN[t, :], :]
            thetas_ND += counts_NSD[:, s, :]
            thetas_ND /= numpy.sum(thetas_ND, axis = -1)[..., None]

            assert numpy.all(thetas_ND >= -1e8)
            assert numpy.min(thetas_ND) >= 0.0

            thetas_MSD[t * N:(t + 1) * N, s, :] = thetas_ND

            with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                print log_weights_TN[t, :]

        log_weights_M = log_weights_TN.reshape(M)

    return (thetas_MSD, log_weights_M)

def mul_dir_mix_fit(counts):
    counts_NSD = counts
    (N, S, D) = counts_NSD.shape
    K = 64
    T = N * 4

    samples_TSD = numpy.empty((T, S, D), numpy.double)
    alphas_KSD = numpy.empty((K, S, D), numpy.double)
    log_weights_SK = numpy.empty((S, K), numpy.double)
    log_responsibilities_KSN = numpy.empty((K, S, N), numpy.double)
    log_post_weights_KSN = numpy.empty((K, S, N), numpy.double)

    for s in xrange(S):
        print ">>>> ESTIMATING RTDS FOR SOLVER", s

        (
            alphas_KSD[:, s, :],
            log_responsibilities_KSN[:, s, :],
            log_weights_SK[s, :],
            ) = \
            borg.statistics.dcm_mixture_estimate_ml(counts_NSD[:, s, :], K)

        log_post_weights_KSN[:, s, :] = log_responsibilities_KSN[:, s, :]
        log_post_weights_KSN[:, s, :] += log_weights_SK[s, :, None]
        log_post_weights_KSN[:, s, :] -= numpy.logaddexp.reduce(log_post_weights_KSN[:, s, :], axis = 0)

    borg.statistics.assert_log_weights(log_post_weights_KSN, axis = -1)

    for t in xrange(T):
        n = t % N

        for s in xrange(S):
            k = borg.statistics.categorical_rv_log(log_post_weights_KSN[:, s, n])

            samples_TSD[t, s, :] = alphas_KSD[k, s, :]

        samples_TSD[t, ...] += counts_NSD[n, ...]
        samples_TSD[t, ...] /= numpy.sum(samples_TSD[t, ...], axis = -1)[..., None]

    assert numpy.all(samples_TSD >= 0.0)

    return [samples_TSD]

class MulDirMatMixSampler(object):
    def __init__(self, K = 16):
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

                logger.info("recorded sample at Gibbs iteration %i", i)

        return theta_samples

