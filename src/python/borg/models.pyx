# cython: profile=True
"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import scipy.stats
import scikits.learn.linear_model
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

    return logs_NM

cdef class Kernel(object):
    """Kernel function interface."""

    cdef double integrate(self, double x, double v):
        """Integrate over the kernel function."""

cdef class DeltaKernel(Kernel):
    """Delta-function kernel."""

    cdef double integrate(self, double x, double v):
        """Integrate over the kernel function."""

        if x <= v:
            return 0.0
        else:
            return -INFINITY

cdef class TruncatedNormalKernel(Kernel):
    """Truncated Gaussian kernel."""

    cdef double _a
    cdef double _b
    cdef double _sigma

    def __init__(self, a, b, sigma = 1.0):
        """Initialize."""

        self._a = a
        self._b = b
        self._sigma = sigma

    cdef double integrate(self, double x, double v):
        """Integrate over the kernel function."""

        return borg.statistics.truncated_normal_log_cdf(self._a, self._b, x, self._sigma, v)

class KernelModel(object):
    """Kernel density estimation model."""

    def __init__(self, successes, failures, durations, bound, alpha, kernel):
        """Initialize."""

        self._successes_NS = successes
        self._failures_NS = failures
        self._durations_NSR = durations
        self._bound = bound
        self._alpha = alpha
        self._kernel = kernel

    def sample(self, M, B):
        """Sample discrete RTDs from this distribution."""

        cdef Kernel kernel = self._kernel

        cdef int N = self._successes_NS.shape[0]
        cdef int S = self._successes_NS.shape[1]
        cdef int C = B + 1

        cdef numpy.ndarray[int, ndim = 2] successes_NS = self._successes_NS
        cdef numpy.ndarray[int, ndim = 2] failures_NS = self._failures_NS
        cdef numpy.ndarray[double, ndim = 3] durations_NSR = self._durations_NSR
        cdef numpy.ndarray[double, ndim = 3] log_rtds_NSC = numpy.empty((N, S, C), numpy.double)

        cdef double alpha = self._alpha
        cdef double resolution = self._bound / B

        for n in xrange(N):
            for s in xrange(S):
                # fill in bins
                log_rtds_NSC[n, s, -1] = libc.math.log(failures_NS[n, s] + alpha - 1.0)

                for b in xrange(B):
                    log_rtds_NSC[n, s, b] = libc.math.log(alpha - 1.0)

                for r in xrange(successes_NS[n, s]):
                    for b in xrange(B):
                        new_mass = \
                            borg.statistics.log_minus(
                                kernel.integrate(durations_NSR[n, s, r], (b + 1) * resolution),
                                kernel.integrate(durations_NSR[n, s, r], (b + 0) * resolution),
                                )

                        log_rtds_NSC[n, s, b] = borg.statistics.log_plus(log_rtds_NSC[n, s, b], new_mass)

                # then normalize
                normalization = libc.math.log(failures_NS[n, s] + successes_NS[n, s] + C * alpha - C)

                for c in xrange(C):
                    log_rtds_NSC[n, s, c] -= normalization

        return log_rtds_NSC[numpy.random.randint(N, size = M)]

    @staticmethod
    def fit(solver_names, training, kernel, alpha = 1.0 + 1e-8):
        """Fit a kernel-density model."""

        logger.info("fitting kernel-density model")

        (successes_NS, failures_NS, durations_NSR) = training.to_runs_array(solver_names)

        bound = training.get_common_budget()

        return KernelModel(successes_NS, failures_NS, durations_NSR, bound, alpha, kernel)

class MultinomialModel(object):
    """Multinomial mixture model."""

    def __init__(self, interval, log_survival, log_weights = None, log_masses = None):
        """Initialize."""

        (N, _, _) = log_survival.shape

        self._interval = interval
        self._log_survival_NSC = log_survival

        if log_weights is None:
            self._log_weights_N = -numpy.ones(N) * numpy.log(N)
        else:
            self._log_weights_N = log_weights

        self._log_masses_NSC = log_masses

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
        """Possible log discrete survival functions."""

        return self._log_masses_NSC

def fit_mul_map(solver_names, training, B):
    """Fit a kernel-density model."""

    logger.info("fitting multinomial mixture model")

    N = len(training.run_lists)
    S = len(solver_names)
    C = B + 1

    outcomes_NSC = training.to_bins_array(solver_names, B)
    attempts_NSC = numpy.sum(outcomes_NSC, axis = -1)[..., None]
    log_components_NSC = numpy.empty((N, S, C), numpy.double)

    ll = 0.0

    for s in xrange(S):
        prior_s = borg.statistics.dcm_estimate_ml(outcomes_NSC[:, s, :])

        # XXX hackish regularization
        prior_s += 1e-32

        # XXX not exactly the correct MAP estimate...
        log_components_NSC[:, s, :] = numpy.log(outcomes_NSC[:, s, :] + prior_s)
        log_components_NSC[:, s, :] -= numpy.log(attempts_NSC[:, s, :] + numpy.sum(prior_s))

        ll += numpy.sum(borg.statistics.dirichlet_log_pdf(prior_s, numpy.exp(log_components_NSC[:, s, :])))

        for n in xrange(N):
            ll += borg.statistics.multinomial_log_pdf(numpy.exp(log_components_NSC[n, s, :]), outcomes_NSC[n, s, :])

    with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
        print numpy.exp(log_components_NSC[:, 0, :])
        #print prior_s

    #print ",".join(map(str, [N, ll]))

    return MultinomialModel(log_components_NSC)

class Mul_ModelFactory(object):
    def sample(self, counts, stored = 4, burn_in = 1024, spacing = 128):
        """Sample parameters of the multinomial model using Gibbs."""

        T = stored
        (N, S, D) = counts.shape

        counts_NSD = numpy.asarray(counts, numpy.intc)
        alpha_D = numpy.ones(D, numpy.double) * 1e-2
        thetas_NSD = numpy.empty((N, S, D), numpy.double)
        theta_samples_TNSD = numpy.empty((T, N, S, D), numpy.double)

        for i in xrange(burn_in + (stored - 1) * spacing + 1):
            # sample multinomial components
            for s in xrange(S):
                for n in xrange(N):
                    thetas_NSD[n, s, :] = numpy.random.dirichlet(alpha_D + counts_NSD[n, s, :])

            # record sample?
            if burn_in <= i and (i - burn_in) % spacing == 0:
                theta_samples_TNSD[int((i - burn_in) / spacing), ...] = thetas_NSD

                logger.info("recorded sample at Gibbs iteration %i", i)

        return theta_samples_TNSD

    def fit(self, solver_names, training, B, T = 1):
        """Fit a model."""

        logger.info("fitting multinomial model")

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

class Mul_Dir_ModelFactory(object):
    def sample(self, counts, stored = 4, burn_in = 1024, spacing = 128):
        """Sample parameters of the DCM model using Gibbs."""

        T = stored
        (N, S, D) = counts.shape

        counts_NSD = numpy.asarray(counts, numpy.intc)
        alpha_SD = numpy.ones((S, D), numpy.double)
        thetas_NSD = numpy.empty((N, S, D), numpy.double)
        theta_samples_TNSD = numpy.empty((T, N, S, D), numpy.double)

        for i in xrange(burn_in + (stored - 1) * spacing + 1):
            # sample multinomial components
            for s in xrange(S):
                for n in xrange(N):
                    thetas_NSD[n, s, :] = numpy.random.dirichlet(alpha_SD[s] + counts_NSD[n, s, :])

                    thetas_NSD[n, s, thetas_NSD[n, s] == 0.0] = 1e-32

            # optimize the Dirichlets
            for s in xrange(S):
                alpha_SD[s, :] = borg.statistics.dirichlet_estimate_map(thetas_NSD[:, s, :], shape = 1.01, scale = 1)

            if burn_in <= i and (i - burn_in) % spacing == 0:
                theta_samples_TNSD[int((i - burn_in) / spacing), ...] = thetas_NSD

                logger.info("recorded sample at Gibbs iteration %i", i)

            assert not numpy.any(alpha_SD < 1e-8)

        return theta_samples_TNSD

    def fit(self, solver_names, training, B, T = 1):
        """Fit a model."""

        logger.info("fitting dirichlet-multinomial model")

        N = len(training.run_lists)
        M = N * T
        S = len(solver_names)
        C = B + 1

        outcomes_NSC = training.to_bins_array(solver_names, B)
        components_MSC = numpy.empty((M, S, C), numpy.double)
        theta_samples_TNSC = self.sample(outcomes_NSC, stored = T)

        for t in xrange(T):
            components_MSC[t * N:(t + 1) * N, ...] = theta_samples_TNSC[t, ...]

            #with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 160, threshold = 1000000):
                #print "T = {0}:".format(t)
                #print theta_samples_TNSC[t, :2, ...]

        log_survival_MSC = borg.statistics.to_log_survival(components_MSC, axis = -1)

        return MultinomialModel(training.get_common_budget() / B, log_survival_MSC, log_masses = numpy.log(components_MSC))

class Mul_DirMix_ModelFactory(object):
    def sample(self, counts, K, stored = 4, burn_in = 1024, spacing = 128):
        """Sample parameters of the DCM mixture model using Gibbs."""

        (N, S, D) = counts.shape
        T = stored

        counts_NSD = numpy.asarray(counts, numpy.intc)
        eta_K = numpy.ones(K, numpy.double)
        pis_SK = numpy.random.dirichlet(numpy.ones(K) + 1e-1, size = S)
        zs_NS = numpy.random.randint(K, size = (N, S))
        alphas_SKD = numpy.ones((S, K, D), numpy.double) + 1e-1
        thetas_NSD = numpy.empty((N, S, D), numpy.double)
        theta_samples_TNSD = numpy.empty((T, N, S, D), numpy.double)

        for i in xrange(burn_in + (stored - 1) * spacing + 1):
            if i % 32 == 0:
                print i

            # sample multinomial components
            for s in xrange(S):
                for n in xrange(N):
                    thetas_NSD[n, s, :] = numpy.random.dirichlet(alphas_SKD[s, zs_NS[n, s]] + counts_NSD[n, s, :])

                    thetas_NSD[n, s, thetas_NSD[n, s] == 0.0] = 1e-32

            # sample cluster assignments
            for s in xrange(S):
                for n in xrange(N):
                    log_posterior_K = numpy.log(numpy.copy(pis_SK[s, :]))

                    for k in xrange(K):
                        log_posterior_K[k] += borg.statistics.dirichlet_log_pdf(alphas_SKD[s, k, :], thetas_NSD[None, n, s, :])

                    log_posterior_K -= numpy.logaddexp.reduce(log_posterior_K)

                    ((zs_NS[n, s],),) = numpy.nonzero(numpy.random.multinomial(1, numpy.exp(log_posterior_K)))

            # optimize the Dirichlets
            for s in xrange(S):
                for k in xrange(K):
                    cluster_thetas_XD = thetas_NSD[numpy.nonzero(zs_NS[:, s] == k)[0], s, :]

                    if cluster_thetas_XD.size > 0:
                        alphas_SKD[s, k, :] = borg.statistics.dirichlet_estimate_map(cluster_thetas_XD, shape = 1.1, scale = 1)
                    else:
                        alphas_SKD[s, k, :] = numpy.random.gamma(1.1, 1, size = D)

            # sample pis
            for s in xrange(S):
                z_counts_K = numpy.zeros(K)

                for n in xrange(N):
                    z_counts_K[zs_NS[n, s]] += 1

                pis_SK[s, :] = numpy.random.dirichlet(eta_K + z_counts_K)


            if burn_in <= i and (i - burn_in) % spacing == 0:
                theta_samples_TNSD[int((i - burn_in) / spacing), ...] = thetas_NSD

                logger.info("recorded sample at Gibbs iteration %i", i)

        return theta_samples_TNSD

    def fit(self, solver_names, training, B, T = 1):
        """Fit a kernel-density model."""

        logger.info("fitting Dirichlet mixture-multinomial model")

        N = len(training.run_lists)
        M = N * T
        S = len(solver_names)
        C = B + 1

        outcomes_NSC = training.to_bins_array(solver_names, B)
        components_MSC = numpy.empty((M, S, C), numpy.double)
        theta_samples_TNSC = self.sample(outcomes_NSC, K = 4, stored = T)

        for t in xrange(T):
            components_MSC[t * N:(t + 1) * N, ...] = theta_samples_TNSC[t, ...]

        log_survival_MSC = borg.statistics.to_log_survival(components_MSC, axis = -1)

        return MultinomialModel(training.get_common_budget() / B, log_survival_MSC, log_masses = numpy.log(components_MSC))

class Mul_DirMatMix_ModelFactory(object):
    def sample(self, counts, K, stored = 4, burn_in = 1024, spacing = 128):
        """Sample parameters of the DCM mixture model using Gibbs."""

        (N, S, D) = counts.shape
        T = stored

        counts_NSD = numpy.asarray(counts, numpy.intc)
        eta_K = numpy.ones(K, numpy.double)
        pi_K = numpy.random.dirichlet(numpy.ones(K) + 1e-1)
        zs_N = numpy.random.randint(K, size = N)
        alphas_SKD = numpy.ones((S, K, D), numpy.double) + 1e-1
        thetas_NSD = numpy.empty((N, S, D), numpy.double)
        theta_samples_TNSD = numpy.empty((T, N, S, D), numpy.double)

        for i in xrange(burn_in + (stored - 1) * spacing + 1):
            # sample multinomial components
            for s in xrange(S):
                for n in xrange(N):
                    thetas_NSD[n, s, :] = numpy.random.dirichlet(alphas_SKD[s, zs_N[n]] + counts_NSD[n, s, :])

                    thetas_NSD[n, s, thetas_NSD[n, s] == 0.0] = 1e-32

            # sample cluster assignments
            for n in xrange(N):
                log_posterior_K = numpy.log(numpy.copy(pi_K))

                for k in xrange(K):
                    for s in xrange(S):
                        log_posterior_K[k] += borg.statistics.dirichlet_log_pdf(alphas_SKD[s, k, :], thetas_NSD[None, n, s, :])

                log_posterior_K -= numpy.logaddexp.reduce(log_posterior_K)

                ((zs_N[n],),) = numpy.nonzero(numpy.random.multinomial(1, numpy.exp(log_posterior_K)))

            # optimize the Dirichlets
            for s in xrange(S):
                for k in xrange(K):
                    cluster_thetas_XD = thetas_NSD[numpy.nonzero(zs_N[:] == k)[0], s, :]

                    if cluster_thetas_XD.size > 0:
                        alphas_SKD[s, k, :] = borg.statistics.dirichlet_estimate_map(cluster_thetas_XD, shape = 1.1, scale = 1)
                    else:
                        alphas_SKD[s, k, :] = numpy.random.gamma(1.1, 1, size = D)

            # sample pis
            z_counts_K = numpy.zeros(K)

            for n in xrange(N):
                z_counts_K[zs_N[n]] += 1

            pi_K[:] = numpy.random.dirichlet(eta_K + z_counts_K)

            # record a sample?
            if burn_in <= i and (i - burn_in) % spacing == 0:
                theta_samples_TNSD[int((i - burn_in) / spacing), ...] = thetas_NSD

                logger.info("recorded sample at Gibbs iteration %i", i)

        return theta_samples_TNSD

    def fit(self, solver_names, training, B, T = 1):
        """Fit a kernel-density model."""

        logger.info("fitting Dirichlet mixture model")

        N = len(training.run_lists)
        M = N * T
        S = len(solver_names)
        C = B + 1

        outcomes_NSC = training.to_bins_array(solver_names, B)
        theta_samples_TNSC = self.sample(outcomes_NSC, K = 4, stored = T)
        components_MSC = numpy.empty((M, S, C), numpy.double)

        for t in xrange(T):
            components_MSC[t * N:(t + 1) * N, ...] = theta_samples_TNSC[t, ...]

        log_survival_MSC = borg.statistics.to_log_survival(components_MSC, axis = -1)

        return MultinomialModel(training.get_common_budget() / B, log_survival_MSC)

