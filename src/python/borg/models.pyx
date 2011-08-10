"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import scipy.stats
import scikits.learn.linear_model
import numpy
import borg
import cargo

cimport cython
cimport libc.math
cimport numpy

logger = cargo.get_logger(__name__, default_level = "DEBUG")

cdef extern from "math.h":
    double INFINITY

def assert_probabilities(array):
    """Assert that an array contains only valid probabilities."""

    assert numpy.all(array >= 0.0)
    assert numpy.all(array <= 1.0)

def assert_log_probabilities(array):
    """Assert that an array contains only valid probabilities."""

    assert numpy.all(array <= 0.0)

def assert_positive_log_probabilities(array):
    """Assert that an array contains only valid positive probabilities."""

    assert numpy.all(array <= 0.0)
    assert numpy.all(array > -numpy.inf)

def assert_weights(array, axis = None):
    """Assert than an array sums to one over a particular axis."""

    assert numpy.all(numpy.abs(numpy.sum(array, axis = axis) - 1.0 ) < 1e-6)

cdef double log_plus(double x, double y):
    """
    Return log(x + y) given log(x) and log(y); see [1].

    [1] Digital Filtering Using Logarithmic Arithmetic. Kingsbury and Rayner, 1970.
    """

    if x == -INFINITY and y == -INFINITY:
        return -INFINITY
    elif x >= y:
        return x + libc.math.log(1.0 + libc.math.exp(y - x))
    else:
        return y + libc.math.log(1.0 + libc.math.exp(x - y))

cdef double log_minus(double x, double y):
    """
    Return log(x - y) given log(x) and log(y); see [1].

    [1] Digital Filtering Using Logarithmic Arithmetic. Kingsbury and Rayner, 1970.
    """

    if x == -INFINITY and y == -INFINITY:
        return -INFINITY
    elif x >= y:
        return x + libc.math.log(1.0 - libc.math.exp(y - x))
    else:
        return y + libc.math.log(1.0 - libc.math.exp(x - y))

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

cdef double log_erf_approximate(double x):
    """Return an approximation to the log of the error function."""

    if x < 0.0:
        return libc.math.NAN

    a = (8.0 * (libc.math.M_PI - 3.0)) / (3.0 * libc.math.M_PI * (4.0 - libc.math.M_PI))
    v = x * x * (4.0 / libc.math.M_PI + a * x * x) / (1.0 + a * x * x)

    return log_minus(0.0, v) / 2.0

cdef double standard_normal_log_pdf(double x):
    """Compute the log of the standard normal PDF."""

    return -(x * x) / 2.0 - libc.math.log(libc.math.M_2_PI) / 2.0

cdef double standard_normal_log_cdf(double x):
    """Compute the log of the standard normal CDF."""

    #if libc.math.abs(x) > 8.0:
        #return log_plus(0.0, log_erf_approximate(x / libc.math.M_SQRT2)) - libc.math.log(2.0)
    #else:
    return libc.math.log((1.0 + libc.math.erf(x / libc.math.M_SQRT2)) / 2.0)

cdef double normal_log_pdf(double mu, double sigma, double x):
    """Compute the log of the normal PDF."""

    cdef double lhs = ((x - mu) * (x - mu)) / (2.0 * sigma * sigma)
    cdef double rhs = libc.math.log(libc.math.M_2_PI * sigma * sigma) / 2.0

    return lhs - rhs

cdef double normal_log_cdf(double mu, double sigma, double x):
    """Compute the log of the normal CDF."""

    cdef double erf_term = libc.math.erf((x - mu) / libc.math.sqrt(2.0 * sigma * sigma))

    return libc.math.log((1.0 + erf_term) / 2.0)

cdef double truncated_normal_log_pdf(double a, double b, double mu, double sigma, double x):
    """Compute the log of the truncated normal PDF."""

    cdef double upper = standard_normal_log_pdf((x - mu) / sigma) - libc.math.log(sigma)
    cdef double lower_lhs = standard_normal_log_cdf((b - mu) / sigma)
    cdef double lower_rhs = standard_normal_log_cdf((a - mu) / sigma)

    return upper - log_minus(lower_lhs, lower_rhs)

cdef double truncated_normal_log_cdf(double a, double b, double mu, double sigma, double x):
    """Compute the log of the truncated normal CDF."""

    cdef double upper_lhs = standard_normal_log_cdf((x - mu) / sigma)
    cdef double upper_rhs = standard_normal_log_cdf((a - mu) / sigma)
    cdef double lower_lhs = standard_normal_log_cdf((b - mu) / sigma)
    cdef double lower_rhs = upper_rhs
    cdef double value = log_minus(upper_lhs, upper_rhs) - log_minus(lower_lhs, lower_rhs)

    #if value == -INFINITY:
        #print "cdf({0}, {1}, {2}) = {3}".format(mu, sigma, x, value)

    return value

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

        return truncated_normal_log_cdf(self._a, self._b, x, self._sigma, v)

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
                            log_minus(
                                kernel.integrate(durations_NSR[n, s, r], (b + 1) * resolution),
                                kernel.integrate(durations_NSR[n, s, r], (b + 0) * resolution),
                                )

                        log_rtds_NSC[n, s, b] = log_plus(log_rtds_NSC[n, s, b], new_mass)

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

def multinomial_log_mass(counts, total_counts, beta):
    """Compute multinomial log probability."""

    assert_probabilities(beta)
    assert_weights(beta, axis = -1)

    log_mass = numpy.sum(counts * numpy.log(beta), axis = -1)
    log_mass += scipy.special.gammaln(total_counts + 1.0)
    log_mass -= numpy.sum(scipy.special.gammaln(counts + 1.0), axis = -1)

    assert_log_probabilities(log_mass)

    return log_mass

def multinomial_log_mass_implied(counts, total_counts, beta):
    """Compute multinomial log probability; final parameter is implied."""

    assert_probabilities(beta)

    implied_p = 1.0 - numpy.sum(beta, axis = -1)
    implied_counts = total_counts - numpy.sum(counts, axis = -1)

    log_mass = numpy.sum(counts * numpy.log(beta), axis = -1)
    log_mass += implied_counts * numpy.log(implied_p)
    log_mass += scipy.special.gammaln(total_counts + 1.0)
    log_mass -= numpy.sum(scipy.special.gammaln(counts + 1.0), axis = -1)
    log_mass -= scipy.special.gammaln(implied_counts + 1.0)

    assert_log_probabilities(log_mass)

    return log_mass

def fit_multinomial_mixture(successes, attempts, K):
    """Fit a multinomial mixture using EM."""

    # mise en place
    (N, B) = successes.shape

    successes_NB = successes
    attempts_N = attempts

    # initialization
    prior_alpha = 1.0 + 1e-2 
    prior_beta = 1.0 + 1e-1 
    prior_upper = prior_alpha - 1.0
    prior_lower = B * prior_alpha + prior_beta - B - 1.0
    initial_n_K = numpy.random.randint(N, size = K)
    components_KB = successes_NB[initial_n_K] + prior_upper
    components_KB /= (attempts_N[initial_n_K] + prior_lower)[:, None]

    # expectation maximization
    old_ll = -numpy.inf

    for i in xrange(512):
        # compute new responsibilities
        log_mass_KN = multinomial_log_mass_implied(successes_NB[None, ...], attempts_N[None, ...], components_KB[:, None, ...])

        log_responsibilities_KN = numpy.copy(log_mass_KN)
        log_responsibilities_KN -= numpy.logaddexp.reduce(log_responsibilities_KN, axis = 0)

        log_weights_K = numpy.logaddexp.reduce(log_responsibilities_KN, axis = 1)
        log_weights_K -= numpy.log(N)

        # compute ll and check for convergence
        ll = numpy.logaddexp.reduce(log_weights_K[:, None] + log_mass_KN, axis = 0)
        ll = numpy.sum(ll)

        logger.debug("log likelihood at EM iteration %i is %f", i, ll)

        if numpy.abs(ll - old_ll) <= 1e-4:
            break

        old_ll = ll

        # compute new components
        responsibilities_KN = numpy.exp(log_responsibilities_KN)

        weighted_successes_KNB = successes_NB[None, ...] * responsibilities_KN[..., None]
        weighted_attempts_KN = attempts_N[None, ...] * responsibilities_KN

        components_KB = numpy.sum(weighted_successes_KNB, axis = 1) + prior_upper
        components_KB /= (numpy.sum(weighted_attempts_KN, axis = 1) + prior_lower)[:, None]

        # split duplicates
        for j in xrange(K):
            for k in xrange(K):
                if j != k and numpy.sum(numpy.abs(components_KB[j] - components_KB[k])) < 1e-6:
                    previous_ll = -numpy.inf
                    n = numpy.random.randint(N)
                    components_KB[k] = successes_NB[n] + prior_upper
                    components_KB[k] /= attempts_N[n] + prior_lower

    assert_probabilities(components_KB)

    return (components_KB, log_weights_K)

def fit_multinomial_matrix_mixture(outcomes, attempts, K):
    """Fit a multinomial matrix mixture using EM."""

    # mise en place
    (N, S, B) = outcomes.shape

    outcomes_NSB = outcomes
    attempts_NS = attempts

    # initialization
    initial_n_K = numpy.arange(N)

    numpy.random.shuffle(initial_n_K)

    initial_n_K = initial_n_K[:K]

    components_KSB = outcomes_NSB[initial_n_K] + 1e-4
    components_KSB /= numpy.sum(components_KSB, axis = -1)[..., None]

    # expectation maximization
    old_ll = -numpy.inf

    for i in xrange(512):
        # compute new responsibilities
        log_mass_KNS = \
            multinomial_log_mass(
                outcomes_NSB[None, ...],
                attempts_NS[None, ...],
                components_KSB[:, None, ...],
                )

        log_responsibilities_KN = numpy.sum(log_mass_KNS, axis = -1)
        log_responsibilities_KN -= numpy.logaddexp.reduce(log_responsibilities_KN, axis = 0)

        log_weights_K = numpy.logaddexp.reduce(log_responsibilities_KN, axis = 1)
        log_weights_K -= numpy.log(N)

        # compute ll and check for convergence
        ll = numpy.logaddexp.reduce(log_weights_K[:, None, None] + log_mass_KNS, axis = 0)
        ll = numpy.sum(ll)

        logger.debug("log likelihood at EM iteration %i is %f", i, ll)

        if numpy.abs(ll - old_ll) <= 1e-4:
            break

        old_ll = ll

        # compute new components
        responsibilities_KN = numpy.exp(log_responsibilities_KN)

        weighted_successes_KNSB = outcomes_NSB[None, ...] * responsibilities_KN[..., None, None]
        weighted_attempts_KNS = attempts_NS[None, ...] * responsibilities_KN[..., None]

        components_KSB = numpy.sum(weighted_successes_KNSB, axis = 1) + 1e-4
        components_KSB /= numpy.sum(components_KSB, axis = -1)[..., None]

        ## split duplicates
        #for j in xrange(K):
            #for k in xrange(K):
                #if j != k and numpy.sum(numpy.abs(components_KSB[j] - components_KSB[k])) < 1e-6:
                    #n = numpy.random.randint(N)

                    #components_KSB[k] = outcomes_NSB[n] + 1e-4
                    #components_KSB[k] /= numpy.sum(components_KSB[k], axis = -1)[..., None]

                    #old_ll = -numpy.inf

    assert_probabilities(components_KSB)
    assert_weights(components_KSB, axis = -1)

    return (components_KSB, log_weights_K)

class MultinomialModel(object):
    """Multinomial mixture model."""

    def __init__(self, log_components_NSC):
        """Initialize."""

        self._log_components_NSC = log_components_NSC

    #def condition(self, failures):
        #"""Return an RTD model conditioned on past runs."""

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

            log_m = log_plus(log_m, log_mn)

        log_m -= libc.math.log(N)

        logs_M[m] = log_m

    return logs_M

named = {
    "kernel": KernelModel,
    "multinomial": MultinomialModel,
    }

