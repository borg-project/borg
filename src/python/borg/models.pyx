"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import scipy.stats
import scikits.learn.linear_model
import numpy
import borg
import cargo

cimport libc.math
cimport numpy
cimport borg.statistics

logger = cargo.get_logger(__name__, default_level = "DEBUG")

cdef extern from "math.h":
    double INFINITY

def sampled_pmfs_log_pmf(pmfs, counts):
    """Compute the log probabilities of instance runs given discrete log PMFs."""

    cdef int N = pmfs.shape[0]
    cdef int S = pmfs.shape[1]
    cdef int C = pmfs.shape[2]
    cdef int M = counts.shape[0]

    cdef numpy.ndarray[double, ndim = 3] pmfs_NSC = pmfs
    cdef numpy.ndarray[int, ndim = 3] counts_MSC = counts
    cdef numpy.ndarray[double, ndim = 1] logs_M = numpy.empty(M, numpy.double)

    cdef int m
    cdef int n
    cdef int s
    cdef int c

    cdef double log_m
    cdef double log_mn
    cdef int counts_msc
    cdef int sum_counts_ms

    for m in xrange(M):
        log_m = -INFINITY

        for n in xrange(N):
            log_mn = 0.0

            for s in xrange(S):
                sum_counts_ms = 0

                for c in xrange(C):
                    counts_msc = counts_MSC[m, s, c]

                    if counts_msc > 0:
                        sum_counts_ms += counts_msc

                        log_mn += counts_msc * pmfs_NSC[n, s, c]
                        log_mn -= libc.math.lgamma(1.0 + counts_msc)

                log_mn += libc.math.lgamma(1.0 + sum_counts_ms)

            log_m = borg.statistics.log_plus(log_m, log_mn)

        log_m -= libc.math.log(N)

        logs_M[m] = log_m

    return logs_M

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

    #def condition(self, failures):
        #"""Return an RTD model conditioned on past runs."""

        #cdef Kernel kernel = self._kernel

        #(N, S) = self._attempts_NS.shape
        #log_weights_N = -numpy.ones(N) * numpy.log(N)

        #for (s, budget) in failures:
            #costs_sNR = self._costs_csr_SNR[s]

            #for n in xrange(N):
                #estimate = 0.0

                #if self._attempts_NS[n, s] > 0:
                    #i = costs_sNR.indptr[n]
                    #j = costs_sNR.indptr[n + 1]

                    #for k in xrange(i, j):
                        #estimate += kernel.integrate(costs_sNR.data[k] - budget)

                    #estimate /= self._attempts_NS[n, s]

                #if estimate < 1.0:
                    #log_weights_N[n] += numpy.log(1.0 - estimate)
                #else:
                    #log_weights_N[n] = -numpy.inf

        #normalization = numpy.logaddexp.reduce(log_weights_N)

        #if normalization > -numpy.inf:
            #log_weights_N -= normalization
        #else:
            ## if nothing applies, revert to our prior
            #log_weights_N = -numpy.ones(N) * numpy.log(N)

        #assert not numpy.any(numpy.isnan(log_weights_N))

        #return KernelPosterior(self, log_weights_N)

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

    def __init__(self, log_components_NSC):
        """Initialize."""

        self._log_components_NSC = log_components_NSC

    def condition(self, failures):
        """Return an RTD model conditioned on past runs."""

        #log_post_weights_K = numpy.copy(self._log_weights_K)
        #components_cdf_KSC = numpy.cumsum(self._components_KSC, axis = -1)

        #for (s, budget) in failures:
            #c = int(budget / self._interval)

            #if c > 0:
                #log_post_weights_K += numpy.log(1.0 - components_cdf_KSC[:, s, c - 1])

        #log_post_weights_K -= numpy.logaddexp.reduce(log_post_weights_K)

        #assert not numpy.any(numpy.isnan(log_post_weights_K))

        #return MultinomialPosterior(self, log_post_weights_K, components_cdf_KSC)

    def sample(self, M, B):
        """Sample discrete RTDs from this distribution."""

        (N, _, C) = self._log_components_NSC.shape

        if B + 1 != C:
            raise ValueError("discretization mismatch")

        return self._log_components_NSC[numpy.random.randint(N, size = M)]

    @property
    def log_components(self):
        """Multinomial components of the model."""

        return self._log_components_NSC

    @staticmethod
    def fit(solver_names, training, B, alpha = 1.0 + 1e-8):
        """Fit a kernel-density model."""

        logger.info("fitting multinomial mixture model")

        C = B + 1

        outcomes_NSC = training.to_bins_array(solver_names, B)
        attempts_NSC = numpy.sum(outcomes_NSC, axis = -1)[..., None]

        log_components_NSC = numpy.log(outcomes_NSC + alpha - 1.0)
        log_components_NSC -= numpy.log(attempts_NSC + C * alpha - C)

        return MultinomialModel(log_components_NSC)

class SolverPriorMultinomialModel(MultinomialModel):
    """Multinomial mixture model with a per-solver prior."""

    @staticmethod
    def fit(solver_names, training, B):
        """Fit a kernel-density model."""

        logger.info("fitting multinomial mixture model")

        N = len(training.run_lists)
        S = len(solver_names)
        C = B + 1

        outcomes_NSC = training.to_bins_array(solver_names, B)
        attempts_NSC = numpy.sum(outcomes_NSC, axis = -1)[..., None]
        log_components_NSC = numpy.empty((N, S, C), numpy.double)

        for s in xrange(S):
            prior_s = borg.statistics.dcm_estimate_ml(outcomes_NSC[:, s, :])

            # XXX hackish regularization
            prior_s += 1e-8

            # XXX not exactly the correct MAP estimate...
            log_components_NSC[:, s, :] = numpy.log(outcomes_NSC[:, s, :] + prior_s)
            log_components_NSC[:, s, :] -= numpy.log(attempts_NSC[:, s, :] + numpy.sum(prior_s))

            #print prior_s
            #print numpy.exp(log_components_NSC[:, s, :])
            #print "..."

        return MultinomialModel(log_components_NSC)

named = {
    "kernel": KernelModel,
    "multinomial": MultinomialModel,
    }

