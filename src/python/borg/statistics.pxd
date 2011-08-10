"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

cimport libc.math

cdef extern from "math.h":
    double NAN
    double INFINITY

cdef double log_plus(double x, double y)
cdef double log_minus(double x, double y)

cdef double log_erf_approximate(double x)
cpdef double digamma(double x) except? -1.0
cdef double inverse_digamma(double x)

cdef double standard_normal_log_pdf(double x)
cdef double standard_normal_log_cdf(double x)

cdef double normal_log_pdf(double mu, double sigma, double x)
cdef double normal_log_cdf(double mu, double sigma, double x)

cdef double truncated_normal_log_pdf(double a, double b, double mu, double sigma, double x)
cdef double truncated_normal_log_cdf(double a, double b, double mu, double sigma, double x)

