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
            self._log_weights_N = -numpy.ones(N) * numpy.log(N)
        else:
            self._log_weights_N = log_weights

        self._log_masses_NSC = log_masses
        self._features = features

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
    def features(self):
        """Features of associates instances."""

        return self._features

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

                    borg.statistics.assert_weights(thetas_NSD[n, s, :], axis = -1)

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
    def __init__(self, K = 16):
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
        #P = 8
        T = N * K
        samples_TSD = numpy.empty((T, S, D), numpy.double)
        log_weights_T = numpy.empty(T, numpy.double)
        features_TF = numpy.empty((T, F), numpy.double)

        for n in xrange(N):
            for k in xrange(K):
                t = n * K + k
                #k = borg.statistics.categorical_rv_log(log_responsibilities_KN[:, n])
                #k = numpy.argmax(log_responsibilities_KN[:, n])

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

class LogNormalMixEstimator(object):
    def __init__(self, int K = 8):
        self._K = K

    def __call__(self, run_data, bins):
        """Fit parameters of the log-normal mixture model."""

        # ...
        (times_SR, ns_SR, failures_NS) = run_data.to_times_arrays()
        budget = run_data.get_common_budget()
        interval = budget / bins

        (N, S) = failures_NS.shape
        D = bins + 1
        K = self._K
        T = N

        # estimate training RTDs
        mus_SK = numpy.empty((S, K), numpy.double)
        sigmas_SK = numpy.empty((S, K), numpy.double)
        thetas_SK = numpy.empty((S, K), numpy.double)
        log_responsibilities_SKN = numpy.empty((S, K, N), numpy.double)

        for s in xrange(S):
            print ">>>> ESTIMATING RTDS FOR SOLVER", s

            (
                mus_SK[s, :],
                sigmas_SK[s, :],
                thetas_SK[s, :],
                log_responsibilities_SKN[s, :],
                ) = \
                borg.statistics.log_normal_mixture_estimate_ml_em(
                    times_SR[s],
                    ns_SR[s],
                    failures_NS[:, s],
                    budget,
                    K,
                    )

            borg.statistics.assert_log_weights(log_responsibilities_SKN[s, :], axis = 0)

        # extract (discrete) samples
        samples_TSD = numpy.zeros((T, S, D), numpy.double)

        for t in xrange(T):
            n = t % N

            for s in xrange(S):
                #k = numpy.argmax(log_responsibilities_SKN[s, :, n])

                for d in xrange(D):
                    for k in xrange(K):
                        below = \
                            borg.statistics.log_normal_log_cdf(
                                mus_SK[s, k],
                                sigmas_SK[s, k],
                                thetas_SK[s, k],
                                d * interval,
                                )

                        if d == D - 1:
                            above = 0.0
                        else:
                            above = \
                                borg.statistics.log_normal_log_cdf(
                                    mus_SK[s, k],
                                    sigmas_SK[s, k],
                                    thetas_SK[s, k],
                                    (d + 1) * interval,
                                    )

                        pi = numpy.exp(log_responsibilities_SKN[s, k, n])

                        samples_TSD[t, s, d] += pi * numpy.exp(borg.statistics.log_minus(above, below))

            with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
                print "INSTANCE", t

                for s in xrange(S):
                    print "~", s, "~", run_data.solver_names[s]
                    print numpy.exp(log_responsibilities_SKN[s, :, n])
                    print times_SR[s][ns_SR[s] == n]
                    print samples_TSD[t, s, :]

            assert numpy.all(samples_TSD[t] >= 0.0)

        #raise SystemExit()

        return \
            MultinomialModel(
                interval,
                borg.statistics.to_log_survival(samples_TSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_TSD),
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

