"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import nose
import numpy
import borg

cimport numpy
cimport borg.statistics

class TestCases(object):
    """Assorted test cases."""

    def test_dirichlet_log_pdf_raw(self):
        def assert_routine_ok(alpha, vector):
            cdef numpy.ndarray[double] alpha_c = alpha
            cdef numpy.ndarray[double] vector_c = vector

            v_slow = borg.statistics.dirichlet_log_pdf(alpha, vector)
            v_raw = borg.statistics.dirichlet_log_pdf_raw(
                alpha.size,
                &alpha_c[0], alpha_c.strides[0],
                &vector_c[0], vector_c.strides[0],
                )

            nose.tools.assert_almost_equal(v_slow, v_raw)

        for i in xrange(8):
            assert_routine_ok(
                numpy.random.rand(16),
                numpy.random.dirichlet(numpy.ones(16)),
                )

    def test_categorical_rv_raw(self):
        def assert_routine_ok(ps):
            S = 65535
            D = ps.size
            counts = numpy.zeros(D)

            cdef numpy.ndarray[double] ps_c = ps

            for _ in xrange(S):
                d = borg.statistics.categorical_rv_raw(D, &ps_c[0], ps_c.strides[0])

                counts[d] += 1

            for d in xrange(D):
                nose.tools.assert_almost_equal(counts[d] / float(S), ps[d], places = 2)

        for i in xrange(8):
            assert_routine_ok(numpy.random.dirichlet(numpy.ones(16)))

    def test_categorical_rv_log_raw(self):
        def assert_routine_ok(ps):
            S = 65535
            D = ps.size
            counts = numpy.zeros(D)

            cdef numpy.ndarray[double] ps_c = ps

            for _ in xrange(S):
                d = borg.statistics.categorical_rv_log_raw(D, &ps_c[0], ps_c.strides[0])

                counts[d] += 1

            for d in xrange(D):
                nose.tools.assert_almost_equal(counts[d] / float(S), numpy.exp(ps[d]), places = 2)

        for i in xrange(8):
            assert_routine_ok(numpy.log(numpy.random.dirichlet(numpy.ones(16))))

