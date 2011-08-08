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

    cdef double log_density(self, double x, double v):
        """Evaluate the kernel function."""

    cdef double integrate(self, double x, double v):
        """Integrate over the kernel function."""

cdef class DeltaKernel(Kernel):
    """Delta-function kernel."""

    cdef double log_density(self, double x, double v):
        """Evaluate the kernel function."""

        if x == v:
            return 0.0
        else:
            return -INFINITY

    cdef double integrate(self, double x, double v):
        """Integrate over the kernel function."""

        if x <= v:
            return 1.0
        else:
            return 0.0

cdef double standard_normal_log_pdf(double x):
    """Compute the log of the standard normal PDF."""

    return -(x * x) / 2.0 - libc.math.log(libc.math.M_2_PI) / 2.0

cdef double standard_normal_log_cdf(double x):
    """Compute the log of the standard normal CDF."""

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

    cdef double log_density(self, double x, double v):
        """Evaluate the kernel function."""

        return truncated_normal_log_pdf(self._a, self._b, x, self._sigma, v)

    cdef double integrate(self, double x, double v):
        """Integrate over the kernel function."""

        raise NotImplementedError()

class KernelModel(object):
    """Kernel density estimation model."""

    def __init__(self, counts, outcomes, success_p, kernel):
        """Initialize."""

        self._counts_NS = counts
        self._outcomes_NSR = outcomes
        self._success_p_NS = success_p
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

    def get_run_log_probabilities(self, test_counts, test_outcomes):
        """Return the log probablities of an array of runs."""

        cdef Kernel kernel = self._kernel

        (N, _) = self._counts_NS.shape
        (M, S, Q) = test_outcomes.shape

        test_counts_MS = test_counts
        test_outcomes_MSQ = test_outcomes

        densities_MSQ = numpy.ones((M, S, Q), numpy.double) * numpy.nan
        total_density = 0.0

        for s in xrange(S):
            for m in xrange(M):
                for q in xrange(test_counts_MS[m, s]):
                    value_msq = -INFINITY
                    outcome_msq = test_outcomes_MSQ[m, s, q]

                    for n in xrange(N):
                        count = self._counts_NS[n, s]

                        if outcome_msq < 0.0:
                            value = libc.math.log(1.0 - self._success_p_NS[n, s])
                        else:
                            value = -INFINITY
                            kernels = 0

                            for r in xrange(count):
                                outcome = self._outcomes_NSR[n, s, r]

                                if outcome >= 0.0:
                                    kernels += 1
                                    log_density = kernel.log_density(outcome, outcome_msq)
                                    value = log_plus(value, log_density)

                            if kernels > 0:
                                value += libc.math.log(self._success_p_NS[n, s]) - libc.math.log(kernels)
                            else:
                                # XXX
                                value = libc.math.log(self._success_p_NS[n, s]) - libc.math.log(6000.0)

                        value_msq = log_plus(value_msq, value)

                    densities_MSQ[m, s, q] = value_msq - libc.math.log(N)

                    total_density += value_msq

        return (total_density, densities_MSQ)

    @staticmethod
    def fit(solver_names, training, kernel, alpha = 1.0 + 1e-6, beta = 1.0 + 1e-6):
        """Fit a kernel-density model."""

        logger.info("fitting kernel-density model")

        (counts_NS, outcomes_NSR) = training.to_array(solver_names)

        failures_NS = numpy.sum(outcomes_NSR < 0.0, axis = -1)
        success_p_NS = (counts_NS - failures_NS + alpha - 1.0) / (counts_NS + alpha + beta - 2.0)

        return KernelModel(counts_NS, outcomes_NSR, success_p_NS, kernel)

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

    def __init__(self, interval, components_KSC, log_weights_K):
        """Initialize."""

        self._interval = interval
        self._components_KSC = components_KSC
        self._log_weights_K = log_weights_K

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

    @staticmethod
    def fit(solver_names, training, cutoff, K, B):
        """Fit a kernel-density model."""

        # XXX just accept a discretization interval; find the max budget from data

        logger.info("fitting multinomial mixture model")

        # mise en place
        C = B + 1
        S = len(solver_names)
        T = len(training.run_lists)

        solver_names = list(solver_names)

        # discretize run data
        interval = cutoff / B
        outcomes_TSC = numpy.zeros((T, S, C))

        for (t, runs) in enumerate(training.run_lists.values()):
            for run in runs:
                s = solver_names.index(run.solver)

                if run.success and run.cost < cutoff:
                    c = int(run.cost / interval)

                    outcomes_TSC[t, s, c] += 1
                else:
                    outcomes_TSC[t, s, B] += 1

        # fit the mixture model
        attempts_TS = numpy.sum(outcomes_TSC, axis = -1)

        (components_KSC, log_weights_K) = \
            fit_multinomial_matrix_mixture(
                outcomes_TSC,
                attempts_TS,
                K,
                )

        return MultinomialModel(interval, components_KSC, log_weights_K)

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

named = {
    "kernel": KernelModel,
    "multinomial": MultinomialModel,
    }

