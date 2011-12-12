#cython: profile=False
"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import sys
import random
import numpy
import scipy.special
import borg

cimport cython
cimport libc.math
cimport libc.limits
cimport numpy

cdef extern from "math.h":
    double NAN
    double INFINITY

logger = borg.get_logger(__name__, default_level = "DEBUG")

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

    assert_probabilities(array)

    assert numpy.all(numpy.abs(numpy.sum(array, axis = axis) - 1.0 ) < 1e-8)

def assert_log_weights(array, axis = None):
    """Assert than an array sums to one over a particular axis."""

    assert_log_probabilities(array)

    assert numpy.all(numpy.abs(numpy.sum(numpy.exp(array), axis = axis) - 1.0) < 1e-8)

def assert_survival(array, axis):
    """Assert that an array contains a (discrete) survival function."""

    assert_probabilities(array)

    lhs = array.swapaxes(0, axis)[1:, ...].swapaxes(0, axis)
    rhs = array.swapaxes(0, axis)[:-1, ...].swapaxes(0, axis)

    assert numpy.all(lhs <= rhs)

def assert_log_survival(array, axis):
    """Assert that an array contains a (discrete) survival function."""

    assert_log_probabilities(array)

    lhs = array.swapaxes(0, axis)[1:, ...].swapaxes(0, axis)
    rhs = array.swapaxes(0, axis)[:-1, ...].swapaxes(0, axis)

    assert numpy.all(lhs <= rhs)

#
# UTILITIES
#

def set_prng_seeds(seed = None):
    """
    Set seeds for all relevant PRNGs.
    
    The (optional) argument is to random.seed(), which is used to initialize
    the other relevant PRNGs. That's statistically iffy, but what isn't?
    """

    random.seed(seed)

    numpy.random.seed(random.randint(0, sys.maxint))

    ii = numpy.iinfo(numpy.int32)

    _set_internal_prng_seed(
        random.randint(ii.min, ii.max),
        random.randint(ii.min, ii.max),
        random.randint(ii.min, ii.max),
        )

@cython.profile(False)
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

def floored_log(array, floor = 1e-64):
    """Return the log of an array, applying a minimum value."""

    copy = numpy.copy(array)

    copy[copy < floor] = floor

    return numpy.log(copy)

def to_log_survival(probabilities, axis):
    """Convert a discrete distribution to log survival-function values."""

    return floored_log(1.0 - numpy.cumsum(probabilities, axis = axis))

def indicator(indices, D, dtype = numpy.intc):
    """Convert a vector of indices to a matrix of indicator vectors."""

    (N,) = indices.shape

    indicator = numpy.zeros((N, D), dtype = dtype)

    for n in xrange(N):
        indicator[n, indices[n]] = 1.0

    return indicator

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

@cython.cdivision(True)
cpdef double digamma(double x):
    """
    Compute the digamma function.

    Implementation adapted from that of Bernardo (1976).
    """

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

@cython.cdivision(True)
cpdef double digamma_approx(double x):
    """Compute an approximation to the digamma function."""

    if x >= 0.6:
        return libc.math.log(x - 0.5)
    else:
        return 0.57721566490153287 - 1.0 / x

@cython.cdivision(True)
cpdef double digamma_approx2(double x):
    """Compute an approximation to the digamma function."""

    return libc.math.log(x + 0.5) - 1.0 / x

@cython.cdivision(True)
cpdef double trigamma(double x):
    """Compute the trigamma function."""

    return scipy.special.polygamma(1, x)

@cython.cdivision(True)
cpdef double trigamma_approx(double x):
    """Compute an approximation to the trigamma function."""

    if x >= 0.6:
        return 1.0 / (x - 0.5)
    else:
        return 1.0 / (x * x)

@cython.infer_types(True)
@cython.cdivision(True)
cpdef double trigamma_approx2(double x):
    """Compute an approximation to the trigamma function."""

    x2 = x * x
    x3 = x2 * x

    return (2.0 * x2 + 2.0 * x + 1.0) / (2.0 * x3 + x2)

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

@cython.infer_types(True)
@cython.cdivision(True)
cpdef double inverse_digamma_newton(double x) except? -1.0:
    """Compute the (numeric) inverse of the digamma function."""

    # initialization
    cdef double y

    if x >= -2.22:
        y = libc.math.exp(x) + 0.5
    else:
        y = -1.0 / (x + 0.57721566490153287)

    # then run Newton-Raphson
    cdef double numerator
    cdef double denominator

    for i in xrange(5):
        numerator = digamma(y) - x
        denominator = trigamma_approx2(y)

        y -= numerator / denominator

    return y

#
# RANDOM VARIATES
#

cdef numpy.int32_t _prng_ix
cdef numpy.int32_t _prng_iy
cdef numpy.int32_t _prng_iz

set_prng_seeds()

cdef void _set_internal_prng_seed(numpy.int32_t ix, numpy.int32_t iy, numpy.int32_t iz):
    """Set the state of the internal PRNG."""

    global _prng_ix
    global _prng_iy
    global _prng_iz

    _prng_ix = (ix % 254) + 1
    _prng_iy = (iy % 254) + 1
    _prng_iz = (iz % 254) + 1

    global _prng_normal_cache_ok

    _prng_normal_cache_ok = False

@cython.infer_types(True)
@cython.cdivision(True)
cpdef double unit_uniform_rv():
    """
    Generate a uniformly-distributed random variate in [0.0,1.0).

    Implements the Wichmann-Hill PRNG.
    """

    global _prng_ix
    global _prng_iy
    global _prng_iz

    _prng_ix = (171 * _prng_ix) % 30269
    _prng_iy = (172 * _prng_iy) % 30307
    _prng_iz = (170 * _prng_iz) % 30323

    u = _prng_ix / 30269.0 + _prng_iy / 30307.0 + _prng_iz / 30323.0

    return u - <numpy.int32_t>u

@cython.infer_types(True)
cdef int categorical_rv_raw(int D, double* ps, int ps_stride):
    """Generate a categorically-distributed random variate."""

    cdef void* ps_p = ps

    u = unit_uniform_rv()

    total = 0.0

    for d in xrange(D - 1):
        total += (<double*>(ps_p + d * ps_stride))[0]

        if total > u:
            return d

    return D - 1

@cython.infer_types(True)
cdef int categorical_rv_log_raw(int D, double* logps, int logps_stride):
    """Generate a categorically-distributed random variate."""

    cdef void* logps_p = logps

    u = unit_uniform_rv()

    total = 0.0

    for d in xrange(D - 1):
        total += libc.math.exp((<double*>(logps_p + d * logps_stride))[0])

        if total > u:
            return d

    return D - 1

cdef double _prng_normal_cache
cdef bint _prng_normal_cache_ok = False

@cython.infer_types(True)
@cython.cdivision(True)
cpdef double unit_normal_rv():
    """
    Generate a (unit) normally-distributed random variate.

    Adapted from Minka.
    """

    # return a value computed previously, if any
    global _prng_normal_cache
    global _prng_normal_cache_ok

    if _prng_normal_cache_ok:
        _prng_normal_cache_ok = False

        return _prng_normal_cache

    # generate a random point inside the unit circle
    cdef double x
    cdef double y
    cdef double radius

    while True:
        x = 2.0 * unit_uniform_rv() - 1.0
        y = 2.0 * unit_uniform_rv() - 1.0

        radius = (x * x) + (y * y)

        if radius < 1.0 and radius != 0.0:
            break

    # Box-Muller formula
    radius = libc.math.sqrt(-2.0 * libc.math.log(radius) / radius)

    x *= radius
    y *= radius

    _prng_normal_cache = y
    _prng_normal_cache_ok = True

    return x

@cython.cdivision(True)
cpdef double unit_gamma_rv(double shape) except? -1.0:
    """
    Generate a gamma-distributed random variate with unit scale.

    See Marsaglia and Tsang, 2000; adapted from Minka.
    """

    assert shape > 0.0

    # boost using Marsaglia's (1961) method
    cdef double boost

    if shape < 1.0:
        boost = libc.math.exp(libc.math.log(unit_uniform_rv()) / shape)
        shape += 1.0
    else:
        boost = 1.0

    # generate the rv
    cdef double d = shape - 1.0 / 3.0
    cdef double c = 1.0 / libc.math.sqrt(9.0 * d)
    cdef double v
    cdef double x
    cdef double u

    while True:
        while True:
            x = unit_normal_rv()
            v = 1.0 + c * x

            if v > 0:
                break

        v = v * v * v
        x = x * x
        u = unit_uniform_rv()

        if (u < 1.0 - 0.0331 * x * x) or (libc.math.log(u) < 0.5 * x + d * (1.0 - v + libc.math.log(v))):
            break

    return boost * d * v

@cython.infer_types(True)
@cython.cdivision(True)
cdef int post_dirichlet_rv(
    unsigned int D,
    double* out,
    unsigned int out_stride,
    double* alphas,
    unsigned int alpha_stride,
    int* counts,
    unsigned int count_stride,
    ) except -1:

    # draw samples from independent gammas
    cdef void* out_p = out
    cdef void* alphas_p = alphas
    cdef void* counts_p = counts

    cdef double alpha
    cdef double count
    cdef double l1_norm = 0.0

    for d in xrange(D):
        alpha = (<double*>(alphas_p + alpha_stride * d))[0]
        count = (<int*>(counts_p + count_stride * d))[0]

        rv = unit_gamma_rv(alpha + count)

        (<double*>(out_p + out_stride * d))[0] = rv

        l1_norm += rv

    # then normalize to the simplex
    for d in xrange(D):
        (<double*>(out_p + out_stride * d))[0] /= l1_norm

    return 0

#
# DISTRIBUTION FUNCTIONS
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

cpdef double multinomial_log_pdf(numpy.ndarray theta, numpy.ndarray counts):
    """Compute the log of the multinomial PDF."""

    cdef int D = theta.shape[0]

    cdef numpy.ndarray[double, ndim = 1] theta_D = theta
    cdef numpy.ndarray[int, ndim = 1] counts_D = counts

    cdef double theta_D_sum = 0.0

    cdef int d

    for d in xrange(D):
        theta_D_sum += theta_D[d]

    cdef double log_pdf = libc.math.lgamma(1.0 + theta_D_sum)

    for d in xrange(D):
        log_pdf += counts_D[d] * libc.math.log(theta_D[d])
        log_pdf -= libc.math.log(1.0 + counts_D[d])

    return log_pdf

cpdef gamma_log_pdf(double x, double shape, double scale):
    """Compute the log of the gamma PDF."""

    return \
        (shape - 1.0) * libc.math.log(x) \
        - libc.math.lgamma(shape) \
        - shape * libc.math.log(scale) \
        - x / scale

def dirichlet_log_pdf(alpha, vectors):
    """Compute the log of the Dirichlet PDF evaluated at multiple vectors."""

    term_a = scipy.special.gammaln(numpy.sum(alpha, axis = -1))
    term_b = numpy.sum(scipy.special.gammaln(alpha), axis = -1)
    term_c = numpy.sum((alpha - 1.0) * numpy.log(vectors), axis = -1)

    return term_a - term_b + term_c

@cython.infer_types(True)
cdef double dirichlet_log_pdf_raw(
    int D,
    double* alpha, int alpha_stride,
    double* vector, int vector_stride,
    ):
    """Compute the log of the Dirichlet PDF evaluated at one vector."""

    cdef void* alpha_p = alpha
    cdef void* vector_p = vector

    # first time
    term_a = 0.0

    for d in xrange(D):
        term_a += (<double*>(alpha_p + alpha_stride * d))[0]

    term_a = libc.math.lgamma(term_a)

    # second term
    term_b = 0.0

    for d in xrange(D):
        term_b += libc.math.lgamma((<double*>(alpha_p + alpha_stride * d))[0])

    # third term
    cdef double alpha_d
    cdef double vector_d

    term_c = 0.0

    for d in xrange(D):
        alpha_d = (<double*>(alpha_p + alpha_stride * d))[0]
        vector_d = (<double*>(vector_p + vector_stride * d))[0]

        term_c += (alpha_d - 1.0) * libc.math.log(vector_d)

    # ...
    return term_a - term_b + term_c

@cython.infer_types(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dirichlet_estimate_ml(vectors):
    """
    Compute the maximum-likelihood Dirichlet distribution.

    Implements Minka's fixed-point iteration.
    """

    cdef int N = vectors.shape[0]
    cdef int D = vectors.shape[1]

    cdef numpy.ndarray[double, ndim = 2] vectors_ND = numpy.asarray(vectors, numpy.double)
    cdef numpy.ndarray[double, ndim = 1] log_pbar_D = numpy.zeros(D, numpy.double)

    # initialization
    cdef double sum_alpha = 0.0
    cdef double digamma_sum_alpha = 0.5772156649015328
    cdef double log_floor = 1e-32

    for d in xrange(D):
        for n in xrange(N):
            log_pbar_D[d] += libc.math.log(vectors_ND[n, d] + log_floor)

        log_pbar_D[d] /= N

        sum_alpha += inverse_digamma_newton(digamma_sum_alpha + log_pbar_D[d])

    digamma_sum_alpha = digamma(sum_alpha)

    # iteration
    cdef double sum_alpha_last = 0.0

    for i in xrange(24):
        sum_alpha = 0.0

        for d in xrange(D):
            sum_alpha += inverse_digamma_newton(digamma_sum_alpha + log_pbar_D[d])

        digamma_sum_alpha = digamma(sum_alpha)

        if libc.math.fabs(1.0 - sum_alpha_last / sum_alpha) < 1e-4:
            break

        sum_alpha_last = sum_alpha

    # termination
    cdef numpy.ndarray[double, ndim = 1] alpha_D = numpy.ones(D, numpy.double) * 1e-1

    for d in xrange(D):
        alpha_D[d] = inverse_digamma_newton(digamma_sum_alpha + log_pbar_D[d])

    return alpha_D

@cython.infer_types(True)
@cython.cdivision(True)
cdef double _inverse_digamma_minus_newton(double x, double t, double N, double c) except? -1.0:
    """Compute a (numeric) inverse for Dirichlet estimation."""

    # approximate the MAP initialization with the (approximate) ML initialization
    cdef double y

    if x >= -2.22:
        y = libc.math.exp(x) + 0.5
    else:
        y = -1.0 / (x + 0.57721566490153287)

    # then run Newton-Raphson
    cdef double numerator
    cdef double denominator

    for i in xrange(32):
        numerator = N * digamma(y) - (c - 1) / y - t - x
        denominator = N * trigamma_approx2(y) + (c - 1) / (y * y)

        y -= numerator / denominator

    return y

@cython.infer_types(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dirichlet_estimate_map(vectors, double shape = 1.0, double scale = 1e8):
    """
    Compute the max a posteriori Dirichlet distribution.

    Implements a version of Minka's fixed-point iteration adapted to
    incorporate a gamma prior.
    """

    cdef int N = vectors.shape[0]
    cdef int D = vectors.shape[1]

    cdef numpy.ndarray[double, ndim = 2] vectors_ND = numpy.asarray(vectors, numpy.double)
    cdef numpy.ndarray[double, ndim = 1] log_pbar_D = numpy.zeros(D, numpy.double)
    cdef numpy.ndarray[double, ndim = 1] alpha_D = numpy.ones(D, numpy.double) * 1e-1

    for d in xrange(D):
        for n in xrange(N):
            log_pbar_D[d] += libc.math.log(vectors_ND[n, d])

    cdef double constant_term
    cdef double alpha_next

    for i in xrange(1024):
        constant_term = 0.0

        for d in xrange(D):
            constant_term += alpha_D[d]

        constant_term = N * digamma_approx2(constant_term) - 1.0 / scale

        alpha_change = 0.0

        for d in xrange(D):
            alpha_next = _inverse_digamma_minus_newton(log_pbar_D[d], constant_term, N, shape)
            alpha_change += libc.math.fabs(alpha_D[d] - alpha_next)
            alpha_D[d] = alpha_next

        if alpha_change < 1e-6:
            return alpha_D

    logger.warning("Dirichlet MAP estimation did not converge; last change in alpha: %s", alpha_change)

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

