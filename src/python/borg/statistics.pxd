"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

cimport libc.math

cdef extern from "math.h":
    double NAN
    double INFINITY

cdef double log_plus(double x, double y)
cdef double log_minus(double x, double y)

cdef double log_erf_approximate(double x)
cpdef double digamma(double x)
cpdef double inverse_digamma(double x)

cdef int categorical_rv_raw(int D, double* logps, int logps_stride)
cdef int categorical_rv_log_raw(int D, double* logps, int logps_stride)

cdef int post_dirichlet_rv(
    unsigned int D,
    double* out,
    unsigned int out_stride,
    double* alphas,
    unsigned int alpha_stride,
    int* counts,
    unsigned int count_stride,
    ) except -1

cdef double standard_normal_log_pdf(double x)
cdef double standard_normal_log_cdf(double x)

cdef double normal_log_pdf(double mu, double sigma, double x)
cdef double normal_log_cdf(double mu, double sigma, double x)

cdef double truncated_normal_log_pdf(double a, double b, double mu, double sigma, double x)
cdef double truncated_normal_log_cdf(double a, double b, double mu, double sigma, double x)

cdef double log_normal_log_pdf(double mu, double sigma, double theta, double x)
cdef double log_normal_log_cdf(double mu, double sigma, double theta, double x)

cdef double binomial_log_pmf(double p, int N, int n)

cdef double dirichlet_log_pdf_raw(
    int D,
    double* alpha, int alpha_stride,
    double* vector, int vector_stride,
    )

