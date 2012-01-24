#cython: profile=False
"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import sys
import random
import numpy
import scipy.special
import scipy.optimize
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
    assert numpy.all(array <= 1.0 + 1e-16)

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

    assert numpy.all(numpy.abs(numpy.sum(array, axis = axis) - 1.0) < 1e-8)

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

@cython.profile(False)
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

    #assert_probabilities(probabilities)
    assert_weights(probabilities, axis = axis)

    return floored_log(1.0 - numpy.cumsum(probabilities, axis = axis))

def indicator(indices, D, dtype = numpy.intc):
    """Convert a vector of indices into a matrix of indicator vectors."""

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

def categorical_rv(ps):
    """Generate a categorically-distributed random variate."""

    (D,) = ps.shape

    cdef numpy.ndarray[double, ndim = 1] ps_D = ps

    return categorical_rv_raw(D, &ps_D[0], ps_D.strides[0])

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

def categorical_rv_log(logps):
    """Generate a categorically-distributed random variate."""

    (D,) = logps.shape

    cdef numpy.ndarray[double, ndim = 1] logps_D = logps

    return categorical_rv_log_raw(D, &logps_D[0], logps_D.strides[0])

@cython.infer_types(True)
cdef int categorical_rv_log_raw(int D, double* logps, int logps_stride):
    """Generate a categorically-distributed random variate."""

    cdef void* logps_p = logps

    u = libc.math.log(unit_uniform_rv())

    total = -INFINITY

    for d in xrange(D - 1):
        total = log_plus(total, (<double*>(logps_p + d * logps_stride))[0])

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

        rv = unit_gamma_rv(alpha + count) + 1e-64 # XXX

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

    return -(x * x) / 2.0 - libc.math.log(2.0 * libc.math.M_PI) / 2.0

cdef double standard_normal_log_cdf(double x):
    """Compute the log of the standard normal CDF."""

    return libc.math.log((1.0 + libc.math.erf(x / libc.math.M_SQRT2)) / 2.0)

cdef double normal_log_pdf(double mu, double sigma, double x):
    """Compute the log of the normal PDF."""

    cdef double lhs = -(x - mu) * (x - mu) / (2.0 * sigma * sigma)
    cdef double rhs = libc.math.log(libc.math.sqrt(2.0 * libc.math.M_PI * sigma * sigma))

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

cdef double log_normal_log_pdf(double mu, double sigma, double theta, double x):
    """Compute the log of the (three-parameter) log-normal PDF."""

    if x <= theta:
        return -INFINITY

    cdef double lhs = -(libc.math.log(x - theta) - mu)**2.0 / (2.0 * sigma * sigma)
    cdef double rhs = libc.math.log((x - theta) * sigma * libc.math.sqrt(2.0 * libc.math.M_PI))

    return lhs - rhs

cdef double log_normal_log_cdf(double mu, double sigma, double theta, double x):
    """Compute the log of the (three-parameter) log-normal CDF."""

    if x <= theta:
        return -INFINITY

    cdef double v = (libc.math.log(x - theta) - mu) / sigma

    return standard_normal_log_cdf(v)

cdef double binomial_log_pmf(double p, int N, int n):
    """Compute the log of the binomial PMF."""

    return \
        libc.math.lgamma(1.0 + N) \
        + n * libc.math.log(p) \
        + (N - n) * libc.math.log(1.0 - p) \
        - libc.math.lgamma(1.0 + n) \
        - libc.math.lgamma(1.0 + N - n)

cdef double multinomial_log_pmf_raw(
    int D,
    double* theta, int theta_stride,
    int* counts, int counts_stride,
    ):
    """Compute the log of the multinomial PDF."""

    cdef void* theta_p = theta
    cdef void* counts_p = counts

    cdef int sum_counts = 0
    cdef int d

    for d in xrange(D):
        sum_counts += (<int*>(counts_p + counts_stride * d))[0]

    cdef double log_pmf = libc.math.lgamma(1.0 + sum_counts)
    cdef double theta_d
    cdef double counts_d

    for d in xrange(D):
        theta_d = (<double*>(theta_p + theta_stride * d))[0]
        counts_d = (<int*>(counts_p + counts_stride * d))[0]

        if theta_d == 0.0 and counts_d > 0:
            return -INFINITY

        log_pmf += counts_d * libc.math.log(theta_d)
        log_pmf -= libc.math.lgamma(1.0 + counts_d)

    return log_pmf

cpdef double multinomial_log_pmf(numpy.ndarray theta, numpy.ndarray counts):
    """Compute the log of the multinomial PDF."""

    cdef numpy.ndarray[double, ndim = 1] theta_D = theta
    cdef numpy.ndarray[int, ndim = 1] counts_D = counts

    return \
        multinomial_log_pmf_raw(
            theta.shape[0],
            &theta_D[0], theta_D.strides[0],
            &counts_D[0], counts_D.strides[0],
            )

cpdef gamma_log_pdf(double x, double shape, double scale):
    """Compute the log of the gamma PDF."""

    return \
        (shape - 1.0) * libc.math.log(x) \
        - libc.math.lgamma(shape) \
        - shape * libc.math.log(scale) \
        - x / scale

def dirichlet_pdf(alpha, vector):
    """Compute the density of a Dirichlet at a vector."""

    product = numpy.prod(vector**(alpha - 1.0))
    normalizer = numpy.prod(scipy.special.gamma(alpha)) / scipy.special.gamma(numpy.sum(alpha))

    return product / normalizer

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

    # first term
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

def dcm_pdf(alpha, vector):
    """Compute the DCM PDF."""

    sum_alpha = numpy.sum(alpha, axis = -1)
    sum_vector = numpy.sum(vector, axis = -1)

    term_l = scipy.special.gamma(sum_alpha) / scipy.special.gamma(sum_alpha + sum_vector)
    term_r = numpy.prod(scipy.special.gamma(vector + alpha) / scipy.special.gamma(alpha), axis = -1)

    return term_l * term_r

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double dcm_log_pdf_raw(
    int D,
    double* alpha, int alpha_stride,
    int* counts, int counts_stride,
    ):
    """Compute the log of the DCM PDF."""

    #cdef numpy.ndarray[double, ndim = 1] alpha_D = alpha
    #cdef numpy.ndarray[int, ndim = 1] counts_D = counts
    cdef void* alpha_p = alpha
    cdef void* counts_p = counts
    cdef double total = 0.0
    cdef int d

    for d in xrange(D):
        total += (<double*>(alpha_p + alpha_stride * d))[0]

    cdef double log_density = libc.math.lgamma(total)

    for d in xrange(D):
        total += (<int*>(counts_p + counts_stride * d))[0]

    log_density -= libc.math.lgamma(total)

    cdef double alpha_d
    cdef int counts_d

    for d in xrange(D):
        alpha_d = (<double*>(alpha_p + alpha_stride * d))[0]
        counts_d = (<int*>(counts_p + counts_stride * d))[0]

        log_density += libc.math.lgamma(counts_d + alpha_d)
        log_density -= libc.math.lgamma(alpha_d)

    return log_density

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

def dcm_estimate_ml_wallach_raw(alpha, counts, weights):
    """
    Compute the maximum-likelihood DCM distribution.

    Implements Wallach's digamma recurrence-relation modification to Minka's
    fixed-point iteration.
    """

    cdef int N = counts.shape[0]
    cdef int D = counts.shape[1]
    cdef int M = numpy.max(counts)
    cdef int L = numpy.max(numpy.sum(counts, axis = -1))

    cdef numpy.ndarray[int, ndim = 2] counts_ND = counts
    cdef numpy.ndarray[double, ndim = 1] alpha_D = alpha
    cdef numpy.ndarray[double, ndim = 1] weights_N = weights
    cdef numpy.ndarray[double, ndim = 1] appearances_L = numpy.zeros(L, numpy.double)
    cdef numpy.ndarray[double, ndim = 2] appearances_MD = numpy.zeros((M, D), numpy.double)

    # compute appearance histograms
    cdef int n
    cdef int d
    cdef int l
    cdef int m

    for n in xrange(N):
        for d in xrange(D):
            m = counts_ND[n, d]

            if m > 0:
                appearances_MD[m - 1, d] += weights_N[n]

        l = 0

        for d in xrange(D):
            l += counts_ND[n, d]

        if l > 0:
            appearances_L[l - 1] += weights_N[n]

    # run through the fixed-point iteration
    cdef double numerator
    cdef double denominator
    cdef double alpha_sum
    cdef double inner_sum
    cdef double change
    cdef double next_alpha_d

    for i in xrange(1024):
        # compute the magnitude of alpha
        alpha_sum = 0.0

        for d in xrange(D):
            alpha_sum += alpha_D[d]

        # compute the update-ratio denominator
        denominator = 0.0
        inner_sum = 0.0

        for l in xrange(L):
            inner_sum += 1.0 / (l + alpha_sum)
            denominator += appearances_L[l] * inner_sum

        # compute the per-dimensional numerators
        change = 0.0

        for d in xrange(D):
            numerator = 0.0
            inner_sum = 0.0

            for m in xrange(M):
                inner_sum += 1.0 / (m + alpha_D[d])
                numerator += appearances_MD[m, d] * inner_sum

            next_alpha_d = alpha_D[d] * numerator / denominator

            if next_alpha_d < 1e-16:
                next_alpha_d = 1e-16

            change += libc.math.fabs(alpha_D[d] - next_alpha_d)
            alpha_D[d] = next_alpha_d

        if change < 1e-10:
            break

def dcm_estimate_ml_wallach(counts, weights = None):
    """Compute the maximum-likelihood DCM distribution."""

    if weights is None:
        weights = numpy.ones(counts.shape[0], numpy.double)

    alpha = numpy.sum(counts, axis = 0, dtype = numpy.double)
    alpha /= numpy.sum(alpha)

    dcm_estimate_ml_wallach_raw(alpha, counts, weights)

    return alpha

@cython.wraparound(False)
@cython.infer_types(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def dcm_mixture_estimate_ml(counts, int K):
    """Fit a DCM mixture using EM."""

    # mise en place
    cdef int N = counts.shape[0]
    cdef int D = counts.shape[1]

    # initialization
    uniques = sorted(set(map(tuple, counts)), key = lambda _: numpy.random.rand())
    components = numpy.empty((K, D))

    for k_ in xrange(K):
        components[k_] = uniques[k_ % len(uniques)]

    components /= numpy.sum(components, axis = -1)[..., None]
    components += 1e-1

    cdef numpy.ndarray[int, ndim = 2] counts_ND = counts
    cdef numpy.ndarray[double, ndim = 2] components_KD = components
    cdef numpy.ndarray[double, ndim = 2] log_densities_KN = numpy.empty((K, N), numpy.double)

    log_weights_K = numpy.zeros(K) - libc.math.log(K)

    # expectation maximization
    cdef unsigned int components_KD_stride1 = components_KD.strides[1]
    cdef unsigned int counts_ND_stride1 = counts_ND.strides[1]

    cdef double previous_ll = -INFINITY

    cdef int i
    cdef int k
    cdef int n

    for i in xrange(128):
        # compute new responsibilities
        for k in xrange(K):
            for n in xrange(N):
                log_densities_KN[k, n] = \
                    dcm_log_pdf_raw(
                        D,
                        &components_KD[k, 0], components_KD_stride1,
                        &counts_ND[n, 0], counts_ND_stride1,
                        )

        log_responsibilities_KN = log_densities_KN + log_weights_K[..., None]
        log_responsibilities_KN -= numpy.logaddexp.reduce(log_responsibilities_KN, axis = 0)

        log_weights_K = numpy.logaddexp.reduce(log_responsibilities_KN, axis = 1)
        log_weights_K -= numpy.log(N)

        # compute ll
        ll_each = numpy.logaddexp.reduce(log_weights_K[:, None] + log_densities_KN, axis = 0)
        ll = numpy.sum(ll_each)

        # check for convergence
        delta_ll = ll - previous_ll
        previous_ll = ll

        if delta_ll >= 0.0:
            logger.debug("ll at EM iteration %i is %f", i, ll)

            if delta_ll <= 1e-8:
                break
        else:
            logger.warning("ll at EM iteration %i is %f <-- DECLINE", i, ll)

        # compute new components
        responsibilities_KN = numpy.exp(log_responsibilities_KN)

        for k in xrange(K):
            dcm_estimate_ml_wallach_raw(components_KD[k], counts_ND, responsibilities_KN[k])

            components_KD[k] += 1e-16

    assert_log_weights(log_responsibilities_KN, axis = 0)

    return (components_KD, log_responsibilities_KN)

@cython.wraparound(False)
@cython.infer_types(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def dcm_matrix_mixture_estimate_ml(counts, int K):
    """Fit a DCM mixture using EM."""

    # mise en place
    cdef int N = counts.shape[0]
    cdef int S = counts.shape[1]
    cdef int D = counts.shape[2]

    # initialization
    counts_strings = map(str, counts)
    counts_dict = dict(zip(counts_strings, counts))
    uniques = sorted(set(counts_strings), key = lambda _: numpy.random.rand())
    components = numpy.empty((K, S, D), numpy.double)

    for k_ in xrange(K):
        components[k_] = counts_dict[uniques[k_ % len(uniques)]]

    components /= numpy.sum(components, axis = -1)[..., None] + 1e-32
    components += 1e-1

    cdef numpy.ndarray[int, ndim = 3] counts_NSD = counts
    cdef numpy.ndarray[double, ndim = 3] components_KSD = components
    cdef numpy.ndarray[double, ndim = 2] log_densities_KN = numpy.empty((K, N), numpy.double)

    log_weights_K = numpy.zeros(K) - libc.math.log(K)

    # expectation maximization
    cdef unsigned int components_KSD_stride2 = components_KSD.strides[2]
    cdef unsigned int counts_NSD_stride2 = counts_NSD.strides[2]

    cdef double previous_ll = -INFINITY

    cdef int i
    cdef int k
    cdef int n
    cdef int s

    for i in xrange(256):
        # compute new responsibilities
        for k in xrange(K):
            for n in xrange(N):
                log_densities_KN[k, n] = 0.0

                for s in xrange(S):
                    log_densities_KN[k, n] += \
                        dcm_log_pdf_raw(
                            D,
                            &components_KSD[k, s, 0], components_KSD_stride2,
                            &counts_NSD[n, s, 0], counts_NSD_stride2,
                            )

        log_responsibilities_KN = log_densities_KN + log_weights_K[..., None]
        log_responsibilities_KN -= numpy.logaddexp.reduce(log_responsibilities_KN, axis = 0)

        log_weights_K = numpy.logaddexp.reduce(log_responsibilities_KN, axis = 1)
        log_weights_K -= numpy.log(N)

        # compute ll
        ll_each = numpy.logaddexp.reduce(log_weights_K[:, None] + log_densities_KN, axis = 0)
        ll = numpy.sum(ll_each)

        # check for convergence
        delta_ll = ll - previous_ll
        previous_ll = ll

        if delta_ll >= 0.0:
            logger.debug("ll at EM iteration %i is %f", i, ll)

            if delta_ll <= 1e-8:
                break
        else:
            logger.warning("ll at EM iteration %i is %f <-- DECLINE", i, ll)

        # compute new components
        responsibilities_KN = numpy.exp(log_responsibilities_KN)

        for k in xrange(K):
            for s in xrange(S):
                dcm_estimate_ml_wallach_raw(components_KSD[k, s, :], counts_NSD[:, s, :], responsibilities_KN[k])

                components_KSD[k, s, :] += 1e-16

    assert numpy.all(numpy.isfinite(components_KSD))
    assert_log_weights(log_responsibilities_KN, axis = 0)

    return (components_KSD, log_responsibilities_KN)

cdef class RightCensoredLogNormalPartial(object):
    """Weighted data l-l under a right-censored log-normal distribution."""

    cdef numpy.ndarray _samples
    cdef numpy.ndarray _ns
    cdef numpy.ndarray _censored
    cdef double _terminus
    cdef numpy.ndarray _log_responsibilities
    cdef int _R
    cdef int _N

    def __init__(self, samples, ns, censored, terminus, log_responsibilities):
        self._samples = samples
        self._ns = ns
        self._censored = censored
        self._terminus = terminus
        self._log_responsibilities = log_responsibilities
        self._R = samples.shape[0]
        self._N = log_responsibilities.shape[0]

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.infer_types(True)
    def objective(self, arguments):
        """Compute the log likelihood."""

        # ...
        cdef double mu = arguments[0]
        cdef double sigma = arguments[1]
        cdef double theta = arguments[2]

        cdef numpy.ndarray[double, ndim = 1] samples_R = self._samples
        cdef numpy.ndarray[int, ndim = 1] ns_R = self._ns
        cdef numpy.ndarray[int, ndim = 1] censored_N = self._censored
        cdef numpy.ndarray[double, ndim = 1] log_responsibilities_N = self._log_responsibilities

        # compute the density
        cdef double log_u = libc.math.log(1e-2)
        cdef double log_1_u = log_minus(0.0, log_u)
        cdef double log_fail_p = log_1_u + log_minus(0.0, log_normal_log_cdf(mu, sigma, theta, self._terminus))

        cdef int r = 0
        cdef int n

        if log_fail_p == -INFINITY:
            log_fail_p = -1e6

        cdef numpy.ndarray[double, ndim = 1] log_densities_N = numpy.zeros(self._N)

        for r in xrange(self._R):
            n = ns_R[r]

            log_densities_N[n] += \
                log_plus(
                    log_u - libc.math.log(self._terminus),
                    log_1_u + log_normal_log_pdf(mu, sigma, theta, samples_R[r] + 1e-8),
                    )

        cdef double log_density = 0.0

        for n in xrange(self._N):
            log_densities_N[n] += censored_N[n] * log_fail_p

            log_density += libc.math.exp(log_responsibilities_N[n]) * log_densities_N[n]

        return -log_density

@cython.wraparound(False)
@cython.infer_types(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def log_normal_mixture_estimate_ml_em(times, ns, censored, double terminus, int K):
    """Fit a right-censored log-normal mixture using EM."""

    # mise en place
    cdef int R = times.shape[0]
    cdef int N = censored.shape[0]

    assert len(times) == len(ns)

    # initialization
    log_terminus = libc.math.log(terminus)

    thetas = [0.0] * (K / 3)
    thetas += [1000.0] * (K / 3)
    thetas += [2000.0] * (K - len(thetas))

    cdef numpy.ndarray[double, ndim = 1] mus_K = numpy.zeros(K)
    cdef numpy.ndarray[double, ndim = 1] sigmas_K = numpy.random.rand(K) * 10.0
    cdef numpy.ndarray[double, ndim = 1] thetas_K = numpy.array(thetas)
    cdef numpy.ndarray[double, ndim = 2] log_densities_KN = numpy.empty((K, N), numpy.double)
    cdef numpy.ndarray[double, ndim = 1] times_R = times + 1.0
    cdef numpy.ndarray[int, ndim = 1] ns_R = numpy.asarray(ns, dtype = numpy.intc)
    cdef numpy.ndarray[int, ndim = 1] censored_N = numpy.asarray(censored, dtype = numpy.intc)

    log_weights_K = numpy.zeros(K) - libc.math.log(K)

    # expectation maximization
    cdef int i
    cdef int k
    cdef int n
    cdef double previous_ll = -INFINITY

    for i in xrange(128):
        print ">>>>", i

        # compute new responsibilities (E step)
        for k in xrange(K):
            log_u = libc.math.log(1e-2)
            log_1_u = log_minus(0.0, log_u)
            log_fail_p = log_1_u + log_minus(0.0, log_normal_log_cdf(mus_K[k], sigmas_K[k], thetas_K[k], terminus))

            print "@", k, "(mu = {0}; sigma = {1}; theta = {2}) [f = {3}] :: {4}".format(mus_K[k], sigmas_K[k], thetas_K[k], numpy.exp(log_fail_p), numpy.exp(log_weights_K[k]))

            if log_fail_p == -INFINITY:
                log_fail_p = -1e6

            for n in xrange(N):
                log_densities_KN[k, n] = log_weights_K[k] + censored_N[n] * log_fail_p

            for r in xrange(R):
                log_densities_KN[k, ns_R[r]] += \
                    log_plus(
                        log_u - log_terminus,
                        log_1_u + log_normal_log_pdf(mus_K[k], sigmas_K[k], thetas_K[k], times_R[r] + 1e-8),
                        )

        log_responsibilities_KN = numpy.copy(log_densities_KN)
        log_responsibilities_KN -= numpy.logaddexp.reduce(log_responsibilities_KN, axis = 0)

        log_weights_K = numpy.logaddexp.reduce(log_responsibilities_KN, axis = 1)
        log_weights_K -= numpy.log(N)

        assert_log_weights(log_weights_K)
        assert_log_weights(log_responsibilities_KN, axis = 0)

        # compute ll and check for convergence
        ll_each = numpy.logaddexp.reduce(log_densities_KN, axis = 0)
        ll = numpy.sum(ll_each)

        delta_ll = ll - previous_ll
        previous_ll = ll

        if delta_ll >= 0.0:
            logger.debug("ll at EM iteration %i is %f", i, ll)

            if delta_ll <= 1e-8:
                break
        else:
            logger.warning("ll at EM iteration %i is %f <-- DECLINE", i, ll)

        assert not numpy.isnan(ll)

        # compute new components (M step)
        for k in xrange(K):
            ll = \
                RightCensoredLogNormalPartial(
                    times_R,
                    ns_R,
                    censored_N,
                    terminus,
                    log_responsibilities_KN[k, :],
                    )
            ((mus_K[k], sigmas_K[k], thetas_K[k]), _, _) = \
                scipy.optimize.fmin_l_bfgs_b(
                    ll.objective,
                    [mus_K[k], sigmas_K[k], thetas_K[k]],
                    approx_grad = True,
                    bounds = [
                        (None, None),
                        (1e-4, None),
                        (0.0, max(0.0, numpy.max(times_R) - 1e-4)),
                        ]
                    )

    return (mus_K, sigmas_K, thetas_K, log_responsibilities_KN)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.infer_types(True)
cdef numpy.ndarray discretize_log_normal(int D, double mu, double sigma, double theta, double terminus):
    """Prepare a discretized distribution."""

    cdef numpy.ndarray[double, ndim = 1] ps_D = numpy.empty(D)
    cdef double interval = terminus / (D - 1)
    cdef double total = 0.0
    cdef int d

    for d in xrange(D):
        below = \
            log_normal_log_cdf(
                mu,
                sigma,
                theta,
                d * interval,
                )

        if d == D - 1:
            above = 0.0
        else:
            above = \
                log_normal_log_cdf(
                    mu,
                    sigma,
                    theta,
                    (d + 1) * interval,
                    )

        ps_D[d] = libc.math.exp(log_minus(above, below)) + 1e-8

        total += ps_D[d]

    for d in xrange(D):
        ps_D[d] /= total

    return ps_D

cdef class DiscreteLogNormalObjective(object):
    """Weighted data l-l under a discretized right-censored log-normal distribution."""

    cdef numpy.ndarray _counts
    cdef double _terminus
    cdef numpy.ndarray _log_responsibilities
    cdef int _N
    cdef int _D

    def __init__(self, counts, terminus, log_responsibilities):
        self._counts = counts
        self._terminus = terminus
        self._log_responsibilities = log_responsibilities
        self._N = counts.shape[0]
        self._D = counts.shape[1]

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.infer_types(True)
    def objective(self, arguments):
        """Compute the log likelihood."""

        cdef double mu = arguments[0]
        cdef double sigma = arguments[1]
        cdef double theta = arguments[2]

        cdef numpy.ndarray[int, ndim = 2] counts_ND = self._counts
        cdef numpy.ndarray[double, ndim = 1] log_responsibilities_N = self._log_responsibilities
        cdef numpy.ndarray[double, ndim = 1] ps_D = \
            discretize_log_normal(
                self._D,
                mu,
                sigma,
                theta,
                self._terminus,
                )

        cdef double ll = 0.0
        cdef int ps_D_stride0 = ps_D.strides[0]
        cdef int counts_ND_stride1 = counts_ND.strides[1]

        for n in xrange(self._N):
            ll += \
                libc.math.exp(log_responsibilities_N[n]) \
                * multinomial_log_pmf_raw(
                    self._D,
                    &ps_D[0], ps_D_stride0,
                    &counts_ND[n, 0], counts_ND_stride1,
                    )

        return -ll

def discrete_log_normal_mixture_estimate_ml(counts, double terminus, int K):
    """Fit a discretized right-censored log-normal mixture using EM."""

    # mise en place
    cdef int N = counts.shape[0]
    cdef int D = counts.shape[1]

    cdef int i
    cdef int k
    cdef int n

    # initialization
    cdef numpy.ndarray[int, ndim = 2] counts_ND = numpy.asarray(counts, dtype = numpy.intc)
    cdef numpy.ndarray[double, ndim = 1] mus_K = numpy.random.rand(K)
    cdef numpy.ndarray[double, ndim = 1] sigmas_K = numpy.random.rand(K)
    cdef numpy.ndarray[double, ndim = 1] thetas_K = numpy.zeros(K)
    cdef numpy.ndarray[double, ndim = 2] ps_KD = numpy.empty((K, D))
    cdef numpy.ndarray[double, ndim = 2] log_densities_KN = numpy.empty((K, N), numpy.double)

    log_responsibilities_KN = numpy.zeros((K, N)) - INFINITY

    for k in xrange(K):
        log_responsibilities_KN[k, numpy.random.randint(N)] = 0.0

    log_weights_K = numpy.zeros(K) - libc.math.log(K)
    log_weights_K -= numpy.logaddexp.reduce(log_weights_K)

    # expectation maximization
    cdef double previous_ll = -INFINITY
    cdef int ps_KD_stride1 = ps_KD.strides[1]
    cdef int counts_ND_stride1 = counts_ND.strides[1]

    for i in xrange(64):
        print ">>>>", i

        # compute new components (M step)
        for k in xrange(K):
            ll = \
                DiscreteLogNormalObjective(
                    counts_ND,
                    terminus,
                    log_responsibilities_KN[k, :],
                    )
            ((mus_K[k], sigmas_K[k], thetas_K[k]), _, _) = \
                scipy.optimize.fmin_l_bfgs_b(
                    ll.objective,
                    [mus_K[k], sigmas_K[k], thetas_K[k]],
                    approx_grad = True,
                    bounds = [
                        (None, None),
                        (1e-4, None),
                        (0.0, terminus),
                        ],
                    )

            #print "@", k, "(mu = {0}; sigma = {1}; theta = {2})".format(mus_K[k], sigmas_K[k], thetas_K[k])

        # compute new responsibilities (E step)
        for k in xrange(K):
            ps_KD[k, :] = discretize_log_normal(D, mus_K[k], sigmas_K[k], thetas_K[k], terminus)

            for n in xrange(N):
                log_densities_KN[k, n] = \
                    multinomial_log_pmf_raw(
                        D,
                        &ps_KD[k, 0], ps_KD_stride1,
                        &counts_ND[n, 0], counts_ND_stride1,
                        )

        log_densities_KN += log_weights_K[..., None]

        log_responsibilities_KN = numpy.copy(log_densities_KN)
        log_responsibilities_KN -= numpy.logaddexp.reduce(log_responsibilities_KN, axis = 0)

        log_weights_K = numpy.logaddexp.reduce(log_responsibilities_KN, axis = 1)
        log_weights_K -= numpy.log(N)

        assert_log_weights(log_weights_K)
        assert_log_weights(log_responsibilities_KN, axis = 0)

        # compute ll and check for convergence
        ll_each = numpy.logaddexp.reduce(log_densities_KN, axis = 0)
        ll = numpy.sum(ll_each)

        delta_ll = ll - previous_ll
        previous_ll = ll

        if delta_ll >= 0.0:
            logger.debug("ll at EM iteration %i is %f", i, ll)

            if delta_ll <= 1e-8:
                break
        else:
            logger.warning("ll at EM iteration %i is %f <-- DECLINE", i, ll)

        assert not numpy.isnan(ll)

    return (ps_KD, log_responsibilities_KN)

