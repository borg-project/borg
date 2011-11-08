"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import nose
import numpy
import borg

cimport numpy
cimport borg.statistics

class TestCase(object):
    def test_dirichlet_log_pdf_raw(self):
        def assert_dirichlet_log_pdf_raw_ok(alpha, vector):
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
            assert_dirichlet_log_pdf_raw_ok(
                numpy.random.rand(16),
                numpy.random.dirichlet(numpy.ones(16)),
                )

