"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import borg

cimport libc.math
cimport cython
cimport numpy
cimport borg.statistics

@cython.infer_types(True)
cdef double kl_divergence(int D, double* left, int left_stride, double* right, int right_stride):
    """Compute the KL divergence between two discrete distributions."""

    cdef void* left_p = left
    cdef void* right_p = right
    cdef double kl = 0.0
    cdef double c = 1e-16 # XXX pseudoprior

    for d in xrange(D):
        p_i = ((<double*>(left_p + left_stride * d))[0] + c) / (1.0 + D * c)
        q_i = ((<double*>(right_p + right_stride * d))[0] + c) / (1.0 + D * c)

        kl += p_i * libc.math.log(p_i / q_i)

    return kl

class KLMeans(object):
    """Bregman clustering (k-means) with KL divergence."""

    def __init__(self, k):
        self._k = k

    @cython.infer_types(True)
    def fit(self, points):
        """Find the cluster centers."""

        points = numpy.asarray(points)

        cdef int N = points.shape[0]
        cdef int C = points.shape[1]
        cdef int D = points.shape[2]
        cdef int K = self._k
        cdef int I = 1024

        cdef numpy.ndarray[double, ndim = 3] points_NCD = points
        cdef numpy.ndarray[double, ndim = 3] centers_KCD = numpy.empty((K, C, D))

        borg.statistics.assert_weights(points_NCD, axis = -1)

        # initialize cluster centers (k-means++)
        cdef numpy.ndarray[double, ndim = 1] divergences_N = numpy.empty(N)
        cdef numpy.ndarray[int, ndim = 1] assignments_N = numpy.empty(N, dtype = numpy.intc)

        cdef int k
        cdef double sum_divergences

        for k in xrange(K):
            # compute distances
            min_divergence = 1.0
            sum_divergences = 0.0

            for n in xrange(N):
                for l in xrange(k):
                    divergence = 0.0

                    for c in xrange(C):
                        divergence += \
                            kl_divergence(
                                D,
                                &centers_KCD[l, c, 0], centers_KCD.strides[2],
                                &points_NCD[n, c, 0], points_NCD.strides[2],
                                )

                    if l == 0 or divergence < min_divergence:
                        min_divergence = divergence

                divergences_N[n] = min_divergence**2.0

                sum_divergences += divergences_N[n]

            # normalize to a distribution, and sample
            for n in xrange(N):
                divergences_N[n] /= sum_divergences

            l = \
                borg.statistics.categorical_rv_raw(
                    N,
                    &divergences_N[0],
                    divergences_N.strides[0],
                    )

            for c in xrange(C):
                for d in xrange(D):
                    centers_KCD[k, c, d] = points_NCD[l, c, d]

        borg.statistics.assert_weights(centers_KCD, axis = -1)

        # iterate k-means
        cdef numpy.ndarray[int, ndim = 1] sizes_K = numpy.empty(K, dtype = numpy.intc)

        for i in xrange(I):
            # assign points to clusters
            changes = 0

            for n in xrange(N):
                min_k = -1
                min_divergence = numpy.inf

                for k in xrange(K):
                    divergence = 0.0

                    for c in xrange(C):
                        divergence += \
                            kl_divergence(
                                D,
                                &centers_KCD[k, c, 0], centers_KCD.strides[2],
                                &points_NCD[n, c, 0], points_NCD.strides[2],
                                )

                    if divergence < min_divergence:
                        min_divergence = divergence
                        min_k = k

                if assignments_N[n] != min_k:
                    changes += 1

                    assignments_N[n] = min_k

            # termination?
            if changes == 0 and i > 0:
                break

            # recompute centers from points
            centers_KCD[:] = 0.0
            sizes_K[:] = 0

            for n in xrange(N):
                k = assignments_N[n]

                for c in xrange(C):
                    for d in xrange(D):
                        centers_KCD[k, c, d] += points_NCD[n, c, d]

                sizes_K[k] += 1

            for k in xrange(K):
                if sizes_K[k] > 0:
                    for c in xrange(C):
                        for d in xrange(D):
                            centers_KCD[k, c, d] /= sizes_K[k]
                else:
                    centers_KCD[k, ...] = points_NCD[numpy.random.randint(N), ...]

        # and we're done
        self._centers = centers_KCD
        self._assignments = assignments_N

        return self

    def predict(self, points):
        """Find the closest cluster center to each point."""

        raise NotImplementedError()

    @property
    def cluster_centers_(self):
        return self._centers

