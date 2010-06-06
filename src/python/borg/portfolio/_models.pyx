"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import  numpy
cimport numpy

cdef class Predictor:
    """
    Core of a performance-tuned model.
    """

    cpdef int predict(self, history, random, out) except -1:
        """
        Make a prediction.
        """

        raise NotImplementedError()

        return -1

cdef extern from "math.h":
    double exp(double x)

cdef extern from "gsl/gsl_errno.h":
    int GSL_SUCCESS

    char* gsl_strerror(int gsl_errno)

cdef extern from "gsl/gsl_sf_result.h":
    ctypedef struct gsl_sf_result:
        double val
        double err

cdef extern from "gsl/gsl_sf.h":
    int gsl_sf_lnpoch_e(double a, double x, gsl_sf_result* result)
    double gsl_sf_lnpoch(double a, double x)

cdef double ln_poch(double a, double x) except? -1:
    """
    Compute the natural log of the Pochhammer function.
    """

    cdef gsl_sf_result result
    cdef int           status = gsl_sf_lnpoch_e(a, x, &result)

    if status != GSL_SUCCESS:
        raise RuntimeError(gsl_strerror(status))

        return -1

    return result.val

cdef class DCM_MixturePredictor(Predictor):
    """
    Core of a performance-tuned DCM mixture model.
    """

    def __init__(self, d_M, pi_K, sum_MK, mix_MKd):
        """
        Initialize.
        """

        # core
        self._d_M     = d_M
        self._pi_K    = pi_K
        self._sum_MK  = sum_MK
        self._mix_MKd = mix_MKd

        # temporaries
        self._post_pi_K    = numpy.empty_like(self._pi_K)
        self._counts_sum_M = numpy.empty_like(self._d_M)

    cpdef int predict(
        self,
        history,
        random,
        out,
        ) except -1:
        """
        Make a prediction.
        """

        # mise en place
        cdef Py_ssize_t M = self._sum_MK.shape[0]
        cdef Py_ssize_t K = self._sum_MK.shape[1]

        cdef numpy.ndarray[unsigned int, ndim = 2] counts_MD    = history
        cdef numpy.ndarray[double, ndim = 2] out_MD             = out
        cdef numpy.ndarray[double, ndim = 1] sum_MK             = self._sum_MK
        cdef numpy.ndarray[double, ndim = 1] mix_MKd            = self._mix_MKd
        cdef numpy.ndarray[double, ndim = 1] post_pi_K          = self._post_pi_K
        cdef numpy.ndarray[unsigned int, ndim = 1] counts_sum_M = self._counts_sum_M

        # calculate per-action vector norms
        cdef Py_ssize_t m
        cdef Py_ssize_t k
        cdef Py_ssize_t d

        for m in xrange(M):
            counts_sum_M[m] = 0

            for k in xrange(K):
                for d in xrange(self._d_M[m]):
                    counts_sum_M[m] += counts_MD[m, d]

        # calculate posterior mixture parameters
        cdef double psigm

        for k in xrange(K):
            post_pi_K[k] = 0.0

            for m in xrange(M):
                psigm = 0.0

                for d in xrange(counts_MD.shape[1]):
                    psigm += gsl_sf_lnpoch(mix_MKd[m, k, d], counts_MD[m, d])

                post_pi_K[k] *= exp(psigm - ln_poch(sum_MK[m, k], counts_sum_M[m]))

        cdef double post_pi_K_sum = 0.0

        for k in xrange(K):
            post_pi_K_sum += post_pi_K[k]

        for k in xrange(K):
            post_pi_K[k] /= post_pi_K_sum

        # calculate outcome probabilities
        cdef double a
        cdef double ll

        for m in xrange(M):
            for d in xrange(self._d_M[m]):
                out_MD[m, d] = 0.0

                for k in xrange(K):
                    a  = mix_MKd[m, k, d] + counts_MD[m, d]
                    ll = gsl_sf_lnpoch(a, 1.0) - ln_poch(counts_sum_M[m], 1.0)

                    out_MD[m, d] += post_pi_K[k] * exp(ll)





# log an outcome-probability table
# rows = {}

# for action in self._actions:
#     ps = rows.get(action.solver, [])

#     ps.append((action.cost, out[action_indices[action]]))

#     rows[action.solver] = ps

# sorted_rows = [(k.name, sorted(v, key = lambda (c, p): c)) for (k, v) in rows.items()]
# sorted_all  = sorted(sorted_rows, key = lambda (k, v): k)
# longest     = max(len(s) for (s, _) in sorted_all)
# table       = \
#     "\n".join(
#         "%s: %s" % (s.ljust(longest + 1), " ".join("%.4f" % p[0] for (c, p) in r)) \
#         for (s, r) in sorted_all \
#         )

# log.debug("probabilities of action success (DCM model):\n%s", table)

