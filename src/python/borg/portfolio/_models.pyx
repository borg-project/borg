"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

# FIXME don't leak memory

import  numpy
cimport numpy

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t

    void* malloc(size_t size)
    void  free  (void *ptr)

cdef class Predictor:
    """
    Core of a performance-tuned model.
    """

    def predict(self, history, out):
        """
        Make a prediction.
        """

        cdef numpy.ndarray[unsigned int, ndim = 1, mode = "c"] history_ = history
        cdef numpy.ndarray[double, ndim = 2, mode = "c"]       out_     = out

        return self.predict_raw(<unsigned int*>history_.data, <double*>out_.data)

    cdef int predict_raw(self, unsigned int* history, double* out) except -1:
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

cdef double ln_poch(double a, double x):
    """
    Compute the natural log of the Pochhammer function.
    """

    cdef gsl_sf_result result
    cdef int           status = gsl_sf_lnpoch_e(a, x, &result)

    if status != GSL_SUCCESS:
        raise RuntimeError("%s (a = %f; x = %f)" % (gsl_strerror(status), a, x))

        return -1

    return result.val

cdef class DCM_MixturePredictor(Predictor):
    """
    Core of a performance-tuned DCM mixture model.
    """

    cdef unsigned int  _M
    cdef unsigned int  _K
    cdef unsigned int  _D
    cdef double*       _pi_K
    cdef double*       _sum_MK
    cdef double*       _mix_MKD
    cdef double*       _post_pi_K
    cdef unsigned int* _counts_sum_M

    def __init__(self, actions, mixture):
        """
        Initialize.
        """

        # cache mixture components appropriately
        self._M = M = mixture.ndomains
        self._K = K = mixture.ncomponents
        self._D = D = len(actions[0].outcomes)

        for action in actions:
            assert len(action.outcomes) == self._D

        cdef numpy.ndarray[double, ndim = 1, mode = "c"] mixture_pi = mixture.pi

        self._pi_K    = <double*>mixture_pi.data
        self._sum_MK  = <double*>malloc(M * K * sizeof(double))
        self._mix_MKD = <double*>malloc(M * K * D * sizeof(double))

        for m in xrange(M):
            for k in xrange(K):
                component           = mixture.components[m, k]
                self._sum_MK[m * K + k]  = component.sum_alpha

                for d in xrange(D):
                    self._mix_MKD[m * (K * D) + k * D + d] = component.alpha[d]

        # build persistent temporaries
        self._post_pi_K    = <double*>malloc(K * sizeof(double))
        self._counts_sum_M = <unsigned int*>malloc(M * sizeof(unsigned int))

    cdef int predict_raw(self, unsigned int* counts_MD, double* out_MD):
        """
        Make a prediction.
        """

        # mise en place
        cdef size_t M = self._M
        cdef size_t K = self._K
        cdef size_t D = self._D

        cdef double*       pi_K         = self._pi_K
        cdef double*       sum_MK       = self._sum_MK
        cdef double*       mix_MKD      = self._mix_MKD
        cdef double*       post_pi_K    = self._post_pi_K
        cdef unsigned int* counts_sum_M = self._counts_sum_M

        # calculate per-action vector norms
        cdef Py_ssize_t m
        cdef Py_ssize_t k
        cdef Py_ssize_t d

        for m in xrange(M):
            counts_sum_M[m] = 0

            for d in xrange(D):
                counts_sum_M[m] += counts_MD[m * D + d]

        # calculate posterior mixture parameters
        cdef double psigm

        for k in xrange(K):
            post_pi_K[k] = pi_K[k]

            for m in xrange(M):
                psigm = 0.0

                for d in xrange(D):
                    psigm += ln_poch(mix_MKD[m * (K * D) + k * D + d], counts_MD[m * D + d])

                post_pi_K[k] *= exp(psigm - ln_poch(sum_MK[m * K + k], counts_sum_M[m]))

        cdef double post_pi_K_sum = 0.0

        for k in xrange(K):
            post_pi_K_sum += post_pi_K[k]

        for k in xrange(K):
            post_pi_K[k] /= post_pi_K_sum

        # calculate outcome probabilities
        cdef double a
        cdef double sum_a
        cdef double ll

        for m in xrange(M):
            for d in xrange(D):
                out_MD[m * D + d] = 0.0

                for k in xrange(K):
                    a     = mix_MKD[m * (K * D) + k * D + d] + counts_MD[m * D + d]
                    sum_a = sum_MK[m * K + k] + counts_sum_M[m]
                    ll    = ln_poch(a, 1.0) - ln_poch(sum_a, 1.0)

                    out_MD[m * D + d] += post_pi_K[k] * exp(ll)

        # success
        return 0





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

