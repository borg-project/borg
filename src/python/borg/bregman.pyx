"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import borg

cimport libc.math
cimport cython
cimport numpy
cimport borg.statistics

logger = borg.get_logger(__name__, default_level = "DEBUG")

cdef extern from "math.h":
    double INFINITY

@cython.cdivision(True)
@cython.infer_types(True)
cdef double kl_divergence_raw(int D, double* left, int left_stride, double* right, int right_stride):
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

@cython.wraparound(False)
@cython.boundscheck(False)
def kl_divergences_all(rtds):
    cdef int N
    cdef int S
    cdef int D

    (N, S, D) = rtds.shape

    cdef numpy.ndarray[double, ndim = 3] rtds_NSD = rtds
    cdef numpy.ndarray[double, ndim = 2] distances_NN = numpy.zeros((N, N))

    cdef int rtds_NSD_stride2 = rtds_NSD.strides[2]
    cdef int n
    cdef int m
    cdef int s

    for n in xrange(N):
        print n
        for m in xrange(N):
            for s in xrange(S):
                distances_NN[n, m] += \
                    kl_divergence_raw(
                        D,
                        &rtds_NSD[n, s, 0], rtds_NSD_stride2,
                        &rtds_NSD[m, s, 0], rtds_NSD_stride2,
                        )

    return distances_NN

@cython.wraparound(False)
@cython.boundscheck(False)
def survival_distances(survival, others):
    cdef int N
    cdef int S
    cdef int D

    (N, S, D) = others.shape

    assert survival.shape == (S, D)

    cdef numpy.ndarray[double, ndim = 2] survival_SD = survival
    cdef numpy.ndarray[double, ndim = 3] others_NSD = others
    cdef numpy.ndarray[double, ndim = 1] distances_N = numpy.empty(N)

    cdef double distance
    cdef double max_distance
    cdef double mean_max_distance
    cdef int n
    cdef int s
    cdef int d

    for n in xrange(N):
        mean_max_distance = 0.0

        for s in xrange(S):
            max_distance = 0.0

            for d in xrange(D):
                distance = libc.math.fabs(survival_SD[s, d] - others_NSD[n, s, d])

                if distance > max_distance:
                    max_distance = distance

            mean_max_distance += max_distance

        mean_max_distance /= S

        distances_N[n] = mean_max_distance

    return distances_N

@cython.wraparound(False)
@cython.boundscheck(False)
def survival_distances_all(survivals):
    (N, S, D) = survivals.shape

    distances_NN = numpy.empty((N, N))

    for n in xrange(N):
        distances_NN[n, :] = survival_distances(survivals[n, :, :], survivals)

    return distances_NN

class KLMeans(object):
    """Bregman clustering (k-means) with KL divergence."""

    def __init__(self, k):
        self._k = k

    @cython.infer_types(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
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
        logger.info("initializing %i cluster centers with k-means++", K)

        cdef numpy.ndarray[double, ndim = 1] divergences_N = numpy.empty(N)
        cdef numpy.ndarray[int, ndim = 1] assignments_N = numpy.empty(N, dtype = numpy.intc)

        cdef int k
        cdef double min_divergence
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
                            kl_divergence_raw(
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
            logger.info("running iteration %i of k-means", i + 1)

            # assign points to clusters
            changes = 0

            for n in xrange(N):
                min_k = -1
                min_divergence = INFINITY

                for k in xrange(K):
                    divergence = 0.0

                    for c in xrange(C):
                        divergence += \
                            kl_divergence_raw(
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

