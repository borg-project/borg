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
        """Return sampled runtime distributions."""

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

    def get_log_probabilities(self, successes, failures):
        """Return the discretized log probablities of runs."""

        cdef Kernel kernel = self._kernel

        cdef int N = self._counts_NS.shape[0]
        cdef int S = self._counts_NS.shape[1]
        cdef int M = successes.shape[0]
        cdef int B = successes.shape[2]

        cdef numpy.ndarray[int, ndim = 2] counts_NS = self._counts_NS
        cdef numpy.ndarray[double, ndim = 2] success_p_NS = self._success_p_NS
        cdef numpy.ndarray[double, ndim = 3] outcomes_NSR = self._outcomes_NSR
        cdef numpy.ndarray[int, ndim = 3] successes_MSB = successes
        cdef numpy.ndarray[int, ndim = 2] failures_MS = failures
        cdef numpy.ndarray[double, ndim = 1] logs_M = numpy.empty(M, numpy.double)

        cdef int s
        cdef int m
        cdef int n
        cdef int b
        cdef int r

        cdef int count
        cdef int kernels
        cdef double bound = self._bound
        cdef double outcome
        cdef double log_ms
        cdef double log_msn
        cdef double bin_lower
        cdef double bin_upper
        cdef double log_cdf_lower
        cdef double log_cdf_upper
        cdef double log_cumulative

        #print self._counts_NS
        #print self._success_p_NS
        #print self._outcomes_NSR

        for m in xrange(M):
            log_m = -INFINITY

            for n in xrange(N):
                log_mn = 0.0

                for s in xrange(S):
                    # score any failures
                    count = failures_MS[m, s]

                    log_mn += failures_MS[m, s] * libc.math.log(1.0 - success_p_NS[n, s])
                    log_mn -= libc.math.lgamma(1.0 + failures_MS[m, s])

                    # score any successes
                    for b in xrange(B):
                        if successes_MSB[m, s, b] == 0:
                            continue

                        count += successes_MSB[m, s, b]

                        bin_lower = (bound / B) * b
                        bin_upper = (bound / B) * (b + 1)

                        log_cdf_lower = -INFINITY
                        log_cdf_upper = libc.math.log(1e-2 / bound)

                        kernels = 0

                        for r in xrange(counts_NS[n, s]):
                            outcome = outcomes_NSR[n, s, r]

                            if outcome >= 0.0:
                                kernels += 1

                                log_cdf_lower = log_plus(log_cdf_lower, kernel.integrate(outcome, bin_lower))
                                log_cdf_upper = log_plus(log_cdf_upper, kernel.integrate(outcome, bin_upper))

                        log_cumulative = log_minus(log_cdf_upper, log_cdf_lower)
                        log_cumulative += libc.math.log(success_p_NS[n, s])
                        log_cumulative -= libc.math.log(kernels + B * 1e-2 / bound)

                        log_mn += successes_MSB[m, s, b] * log_cumulative
                        log_mn -= libc.math.lgamma(1.0 + successes_MSB[m, s, b])

                    log_mn += libc.math.lgamma(1.0 + count)

                log_m = log_plus(log_m, log_mn)

            log_m -= libc.math.log(N)

            logs_M[m] = log_m

        assert_positive_log_probabilities(logs_M)

        return logs_M

    @staticmethod
    def fit(solver_names, training, kernel, alpha = 1.0 + 1e-8):
        """Fit a kernel-density model."""

        logger.info("fitting kernel-density model")

        (successes_NS, failures_NS, durations_NSR) = training.to_runs_array(solver_names)

        bound = training.get_common_budget()

        return KernelModel(successes_NS, failures_NS, durations_NSR, bound, alpha, kernel)

cdef class Posterior(object):
    pass

cdef class KernelPosterior(Posterior):
    """Conditioned kernel density model."""

    cdef object _model
    cdef object _log_weights_N

    def __init__(self, model, log_weights_N):
        """Initialize."""

        self._model = model
        self._log_weights_N = log_weights_N

    @property
    def components(self):
        """The number of mixture components."""

        return self._log_weights_N.size

    def get_weights(self):
        """The weights of mixture components in the model."""

        return self._log_weights_N

    def get_log_cdf_array(self, budgets):
        """Compute the log CDF."""

        cdef Kernel kernel = self._model._kernel

        cdef int N = self._model._attempts_NS.shape[0]
        cdef int S = self._model._attempts_NS.shape[1]
        cdef int B = budgets.shape[0]

        cdef numpy.ndarray[long, ndim = 2] attempts_NS = self._model._attempts_NS
        cdef numpy.ndarray[double, ndim = 1] budgets_B = budgets
        cdef numpy.ndarray[double, ndim = 3] log_cdf_NSB = numpy.empty((N, S, B))

        cdef int i
        cdef int j
        cdef int k
        cdef int n
        cdef double estimate = 0.0

        cdef numpy.ndarray[int] costs_csr_sNR_indptr
        cdef numpy.ndarray[double] costs_csr_sNR_data

        for s in xrange(S):
            costs_csr_sNR = self._model._costs_csr_SNR[s]

            if costs_csr_sNR is None:
                log_cdf_NSB[:, s, :] = -INFINITY
            else:
                costs_csr_sNR_indptr = costs_csr_sNR.indptr
                costs_csr_sNR_data = costs_csr_sNR.data

                for n in xrange(N):
                    i = costs_csr_sNR_indptr[n]
                    j = costs_csr_sNR_indptr[n + 1]

                    for b in xrange(B):
                        estimate = 0.0

                        if attempts_NS[n, s] > 0:
                            for k in xrange(i, j):
                                estimate += kernel.integrate(costs_csr_sNR_data[k], budgets_B[b])

                            estimate /= attempts_NS[n, s]

                        if estimate > 0.0:
                            log_cdf_NSB[n, s, b] = libc.math.log(estimate)
                        else:
                            log_cdf_NSB[n, s, b] = -INFINITY

        return log_cdf_NSB

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

    def __init__(self, interval, budget, components_NSC):
        """Initialize."""

        self._interval = interval
        self._budget = budget
        self._components_NSC = components_NSC

    def condition(self, failures):
        """Return an RTD model conditioned on past runs."""

        log_post_weights_K = numpy.copy(self._log_weights_K)
        components_cdf_KSC = numpy.cumsum(self._components_KSC, axis = -1)

        for (s, budget) in failures:
            c = int(budget / self._interval)

            if c > 0:
                log_post_weights_K += numpy.log(1.0 - components_cdf_KSC[:, s, c - 1])

        log_post_weights_K -= numpy.logaddexp.reduce(log_post_weights_K)

        assert not numpy.any(numpy.isnan(log_post_weights_K))

        return MultinomialPosterior(self, log_post_weights_K, components_cdf_KSC)

    #def get_log_probabilities(self, successes, failures):
        #"""Return the discretized log probablities of runs."""

        #(N, S, C) = self._components_NSC.shape
        #(M, _, D) = successes.shape

        #successes_MSD = successes
        #failures_MS = failures
        #outcomes_MSC = numpy.zeros((M, S, B + 1), numpy.intc)

        #outcomes_MSC[..., -1] = failures_MS

        #d_interval = self._budget / D

        #for d in xrange(D):
            #c = int((d + 1) * d_interval / self._interval)

        #return logs_M

    @staticmethod
    def fit(solver_names, training, B):
        """Fit a kernel-density model."""

        logger.info("fitting multinomial mixture model")

        # mise en place
        C = B + 1
        S = len(solver_names)
        N = len(training.run_lists)

        solver_names = list(solver_names)

        # discretize run data
        common_budget = training.get_common_budget()
        interval = common_budget / B
        outcomes_NSC = numpy.zeros((N, S, C))

        for (t, runs) in enumerate(training.run_lists.values()):
            for run in runs:
                s = solver_names.index(run.solver)

                if run.success and run.cost < common_budget:
                    c = int(run.cost / interval)

                    outcomes_NSC[t, s, c] += 1
                else:
                    outcomes_NSC[t, s, B] += 1

        # fit the mixture model
        attempts_NS = numpy.sum(outcomes_NSC, axis = -1)
        components_NSC = outcomes_NSC / attempts_NS[..., None]

        return MultinomialModel(interval, common_budget, components_NSC)

cdef class MultinomialPosterior(Posterior):
    """Conditioned multinomial mixture model."""

    cdef object _model
    cdef object _log_weights_K
    cdef object _components_cdf_KSC

    def __init__(self, model, log_weights_K, components_cdf_KSC):
        """Initialize."""

        self._model = model
        self._log_weights_K = log_weights_K
        self._components_cdf_KSC = numpy.cumsum(model._components_KSC, axis = -1)

    @property
    def components(self):
        """The number of mixture components."""

        return self._log_weights_K.size

    def get_weights(self):
        """The weights of mixture components in the model."""

        return self._log_weights_K

    def get_log_cdf_array(self, budgets):
        """Compute the log CDF."""

        budgets_B = budgets

        K = self._components_cdf_KSC.shape[0]
        S = self._components_cdf_KSC.shape[1]
        B = budgets.shape[0]

        if B == 0:
            return numpy.empty((K, S, B))
        else:
            indices_B = numpy.array(budgets_B / self._model._interval, numpy.intc)

            log_cdf_KSB = numpy.empty((K, S, B))

            log_cdf_KSB[:, :, indices_B == 0] = -numpy.inf
            log_cdf_KSB[:, :, indices_B > 0] = numpy.log(self._components_cdf_KSC[:, :, indices_B[indices_B > 0] - 1])

            return log_cdf_KSB

def sampled_pmfs_log_pmf(pmfs, counts):
    """Compute the log probabilities of instance runs given discrete CDFs."""

    pmfs_NSC = pmfs
    counts_MSC = counts

    (N, S, C) = pmfs_NSC.shape
    (M, _, _) = counts_MSC.shape

    logs_M = numpy.empty(M, numpy.double)

    for m in xrange(M):
        log_m = -INFINITY

        for n in xrange(N):
            log_mn = 0.0

            for s in xrange(S):
                sum_counts_ms = 0

                for c in xrange(C):
                    counts_msc = counts_MSC[m, s, c]

                    sum_counts_ms += counts_msc

                    log_mn += counts_msc * libc.math.log(pmfs_NSC[n, s, c])
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

