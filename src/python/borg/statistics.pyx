"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import scipy.special
import cargo

cimport libc.math
cimport numpy

logger = cargo.get_logger(__name__, default_level = "DEBUG")

cdef extern from "math.h":
    double NAN
    double INFINITY

#
# ASSERTIONS
#

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

#
# UTILITIES
#

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

#
# SPECIAL FUNCTIONS
#

cdef double log_erf_approximate(double x):
    """Return an approximation to the log of the error function."""

    if x < 0.0:
        return libc.math.NAN

    a = (8.0 * (libc.math.M_PI - 3.0)) / (3.0 * libc.math.M_PI * (4.0 - libc.math.M_PI))
    v = x * x * (4.0 / libc.math.M_PI + a * x * x) / (1.0 + a * x * x)

    return log_minus(0.0, v) / 2.0

cpdef double digamma(double x) except? -1.0:
    """
    Compute the digamma function.

    Implementation adapted from that of Bernardo (1976).
    """

    if x < 0.0:
        raise ValueError("negative x passed to digamma()")

    cdef double s = 1e-5
    cdef double c = 8.5
    cdef double s3 = 8.333333333e-2
    cdef double s4 = 8.333333333e-3
    cdef double s5 = 3.968253968e-3
    cdef double d1 = -0.5772156649

    cdef double r
    cdef double y
    cdef double v

    if x > s:
        y = x
        v = 0.0

        while y < c:
            v -= 1.0 / y
            y += 1.0

        r = 1.0 / y
        v += libc.math.log(y) - r / 2.0
        r = 1.0 / (y * y)
        v -= r * (s3 - r * (s4 - r * s5))
    else:
        v = d1 - 1.0 / x

    return v

cpdef double inverse_digamma(double x):
    """Compute the (numeric) inverse of the digamma function."""

    if x == -INFINITY:
        return 0.0

    cdef double y = libc.math.exp(x)
    cdef double d = 1.0

    while d > 1e-16:
        if digamma(y) < x:
            y += d
        else:
            y -= d

        d /= 2.0

    return y

#
# DISTRIBUTIONS
#

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

cdef double truncated_normal_log_cdf(double a, double b, double mu, double sigma, double x):
    """Compute the log of the truncated normal CDF."""

    cdef double upper_lhs = standard_normal_log_cdf((x - mu) / sigma)
    cdef double upper_rhs = standard_normal_log_cdf((a - mu) / sigma)
    cdef double lower_lhs = standard_normal_log_cdf((b - mu) / sigma)
    cdef double lower_rhs = upper_rhs

    return log_minus(upper_lhs, upper_rhs) - log_minus(lower_lhs, lower_rhs)

cpdef gamma_log_pdf(double x, double shape, double scale):
    """Compute the log of the gamma PDF."""

    return \
        (shape - 1.0) * libc.math.log(x) \
        - libc.math.lgamma(shape) \
        - shape * libc.math.log(scale) \
        - x / scale

def dirichlet_log_pdf(alpha, vectors):
    """Compute the log of the Dirichlet PDF."""

    term_a = scipy.special.gammaln(numpy.sum(alpha, axis = -1))
    term_b = numpy.sum(scipy.special.gammaln(alpha), axis = -1)
    term_c = numpy.sum((alpha - 1.0) * numpy.log(vectors), axis = -1)

    return term_a - term_b + term_c

cdef double _inverse_digamma_minus(double x, double N, double c) except? -1.0:
    """Compute a (numeric) inverse for Dirichlet estimation."""

    cdef double y = libc.math.exp(x / N)
    cdef double d = 1.0
    cdef double v

    if y == 0.0:
        return 0.0

    while d > 1e-16:
        v = N * digamma(y) - (c - 1.0) / y

        if v < x:
            y += d
        else:
            y -= d

        d /= 2.0

    if libc.math.fabs(v - x) > 1e-8:
        logger.warning(
            "modified-digamma inversion failed to converge: f(%f, %f, %f) = %f != %f",
            y,
            N,
            c,
            v,
            x,
            )

    return y

def dirichlet_estimate_map(vectors, shape = 1.0, scale = 1e8):
    """
    Compute the maximum-likelihood Dirichlet distribution.

    Implements a version of Minka's fixed-point iteration adapted to
    incorporate a gamma prior.
    """

    (N, D) = vectors.shape

    vectors_ND = numpy.asarray(vectors, numpy.double)
    log_pbar_D = numpy.zeros(D, numpy.double)
    alpha_D = numpy.ones(D, numpy.double) * 1e-1

    for d in xrange(D):
        for n in xrange(N):
            log_pbar_D[d] += libc.math.log(vectors_ND[n, d])

        log_pbar_D[d] /= N

    for i in xrange(32768):
        last_alpha_D = numpy.copy(alpha_D)
        psi_alpha_D = N * digamma(numpy.sum(alpha_D)) + N * log_pbar_D - 1.0 / scale

        for d in xrange(D):
            alpha_D[d] = _inverse_digamma_minus(psi_alpha_D[d], N, shape)

        if numpy.sum(numpy.abs(alpha_D - last_alpha_D)) <= 1e-16:
            return alpha_D

    logger.warning("Dirichlet MAP estimation terminated before convergence")

    return alpha_D

def dcm_pdf(vector, alpha):
    """Compute the DCM PDF."""

    sum_alpha = numpy.sum(alpha, axis = -1)
    sum_vector = numpy.sum(vector, axis = -1)

    term_l = scipy.special.gamma(sum_alpha) / scipy.special.gamma(sum_alpha + sum_vector)
    term_r = numpy.prod(scipy.special.gamma(vector + alpha) / scipy.special.gamma(alpha), axis = -1)

    return term_l * term_r

def dcm_log_pdf(vector, alpha):
    """Compute the log of the DCM PDF."""

    return numpy.log(dcm_pdf(vector, alpha))

def dcm_estimate_ml(counts, weights = None):
    """
    Compute the maximum-likelihood DCM distribution.

    Implements Minka's fixed-point iteration.
    """

    cdef int N = counts.shape[0]
    cdef int D = counts.shape[1]

    if weights is None:
        weights = numpy.ones(N, numpy.double)

    alpha = numpy.sum(counts, axis = 0, dtype = numpy.double)
    alpha /= numpy.sum(alpha)

    cdef numpy.ndarray[double, ndim = 2] counts_ND = counts.astype(numpy.double)
    cdef numpy.ndarray[double, ndim = 1] weights_N = weights
    cdef numpy.ndarray[double, ndim = 1] sum_counts_N = numpy.sum(counts_ND, axis = 1)
    cdef numpy.ndarray[double, ndim = 1] alpha_D = alpha

    cdef int i
    cdef int d
    cdef int n

    cdef double total_weight = numpy.sum(weights_N)
    cdef double concentration
    cdef double denominator
    cdef double numerator
    cdef double change

    for i in xrange(1024):
        concentration = 0.0

        for d in xrange(D):
            concentration += alpha_D[d]

        denominator = 0.0

        for n in xrange(N):
            denominator += weights_N[n] * digamma(sum_counts_N[n] + concentration)
            
        denominator -= total_weight * digamma(concentration)

        change = 0.0

        for d in xrange(D):
            if alpha_D[d] > 0.0:
                numerator = 0.0

                for n in xrange(N):
                    numerator += weights_N[n] * digamma(counts_ND[n, d] + alpha_D[d])
                    
                numerator -= total_weight * digamma(alpha_D[d])

                alpha_d = alpha_D[d] * numerator / denominator

                change += abs(alpha_D[d] - alpha_d)

                alpha_D[d] = alpha_d

        if change < 1e-8:
            break

    return alpha_D

def dcm_mixture_estimate_ml(counts, K):
    """Fit a DCM mixture using EM."""

    # mise en place
    (N, D) = counts.shape

    counts_ND = counts

    # initialization
    initial_n_K = numpy.random.randint(N, size = K)

    components_KD = counts_ND[initial_n_K]
    components_KD += 1e-8
    components_KD /= numpy.sum(components_KD, axis = 1)[:, None]

    log_densities_KN = numpy.empty((K, N), numpy.double)

    # expectation maximization
    previous_ll = -numpy.inf

    for i in xrange(512):
        # compute new responsibilities
        for k in xrange(K):
            for n in xrange(N):
                log_densities_KN[k, n] = dcm_log_pdf(components_KD[k], counts_ND[n])

        log_responsibilities_KN = numpy.copy(log_densities_KN)
        log_responsibilities_KN -= numpy.logaddexp.reduce(log_responsibilities_KN, axis = 0)

        log_weights_K = numpy.logaddexp.reduce(log_responsibilities_KN, axis = 1)
        log_weights_K -= numpy.log(N)

        # compute ll and check for convergence
        ll = numpy.logaddexp.reduce(log_weights_K[:, None] + log_densities_KN, axis = 0)
        ll = numpy.sum(ll)

        delta_ll = ll - previous_ll
        previous_ll = ll

        if delta_ll >= 0.0:
            logger.debug("ll change at EM iteration %i is %f", i, ll)

            if delta_ll <= 1e-8:
                break
        else:
            logger.warning("ll change at EM iteration %i is %f <-- DECLINE", i, ll)

        # compute new components
        responsibilities_KN = numpy.exp(log_responsibilities_KN)

        for k in xrange(K):
            components_KD[k] = dcm_estimate_ml(counts_ND, responsibilities_KN[k])

    return components_KD

