"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import  numpy
cimport numpy

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

cdef double ln_poch(double a, double x) except? -1:
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

    cdef _pi_K
    cdef _d_M
    cdef _sum_MK
    cdef _mix_MKd
    cdef _post_pi_K
    cdef _counts_sum_M

    def __init__(self, actions, mixture):
        """
        Initialize.
        """

        # cache mixture components appropriately
        M = mixture.ndomains
        K = mixture.ncomponents

        self._pi_K    = mixture.pi
        self._d_M     = numpy.array([len(a.outcomes) for a in actions], numpy.uint)
        self._sum_MK  = numpy.empty((M, K))
        self._mix_MKd = numpy.empty((M, K, numpy.max(self._d_M)))

        for m in xrange(M):
            for k in xrange(K):
                component           = mixture.components[m, k]
                self._sum_MK[m, k]  = component.sum_alpha
                self._mix_MKd[m, k] = component.alpha

        # build persistent temporaries
        self._post_pi_K    = numpy.empty_like(self._pi_K)
        self._counts_sum_M = numpy.empty_like(self._d_M)

    cdef int predict_raw(self, unsigned int* counts_MD, double* out_MD) except -1:
        """
        Make a prediction.
        """

        # mise en place
        cdef Py_ssize_t M = self._sum_MK.shape[0]
        cdef Py_ssize_t K = self._sum_MK.shape[1]
        cdef Py_ssize_t D = self._mix_MKd.shape[2]

        cdef numpy.ndarray[double, ndim = 1] pi_K               = self._pi_K
        cdef numpy.ndarray[unsigned int, ndim = 1] d_M          = self._d_M
        cdef numpy.ndarray[double, ndim = 2] sum_MK             = self._sum_MK
        cdef numpy.ndarray[double, ndim = 3] mix_MKd            = self._mix_MKd
        cdef numpy.ndarray[double, ndim = 1] post_pi_K          = self._post_pi_K
        cdef numpy.ndarray[unsigned int, ndim = 1] counts_sum_M = self._counts_sum_M

        # calculate per-action vector norms
        cdef Py_ssize_t m
        cdef Py_ssize_t k
        cdef Py_ssize_t d

        for m in xrange(M):
            counts_sum_M[m] = 0

            for d in xrange(d_M[m]):
                counts_sum_M[m] += counts_MD[m * D + d]

        # calculate posterior mixture parameters
        cdef double psigm

        for k in xrange(K):
            post_pi_K[k] = pi_K[k]

            for m in xrange(M):
                psigm = 0.0

                for d in xrange(d_M[m]):
                    psigm += ln_poch(mix_MKd[m, k, d], counts_MD[m * D + d])

                post_pi_K[k] *= exp(psigm - ln_poch(sum_MK[m, k], counts_sum_M[m]))

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
            for d in xrange(d_M[m]):
                out_MD[m * D + d] = 0.0

                for k in xrange(K):
                    a     = mix_MKd[m, k, d] + counts_MD[m * D + d]
                    sum_a = sum_MK[m, k] + counts_sum_M[m]
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

