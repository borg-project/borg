"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import nose
import numpy
import scipy.stats
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

    def test_standard_normal_log_pdf(self):
        def assert_routine_ok(x):
            v_ours = borg.statistics.standard_normal_log_pdf(x)
            v_true = scipy.stats.norm.logpdf(x)

            nose.tools.assert_almost_equal(v_ours, v_true)

        for i in xrange(8):
            assert_routine_ok(numpy.random.normal())

    def test_standard_normal_log_cdf(self):
        def assert_routine_ok(x):
            v_ours = borg.statistics.standard_normal_log_cdf(x)
            v_true = scipy.stats.norm.logcdf(x)

            nose.tools.assert_almost_equal(v_ours, v_true)

        for i in xrange(8):
            assert_routine_ok(numpy.random.normal())

    def test_normal_log_pdf(self):
        def assert_routine_ok(x, mu, sigma):
            v_ours = borg.statistics.normal_log_pdf(mu, sigma, x)
            v_true = scipy.stats.norm.logpdf(x, loc = mu, scale = sigma)

            nose.tools.assert_almost_equal(v_ours, v_true)

        for i in xrange(8):
            assert_routine_ok(
                numpy.random.normal(),
                numpy.random.normal(),
                numpy.random.rand() + 1.0,
                )

    def test_normal_log_cdf(self):
        def assert_routine_ok(x, mu, sigma):
            v_ours = borg.statistics.normal_log_cdf(mu, sigma, x)
            v_true = scipy.stats.norm.logcdf(x, loc = mu, scale = sigma)

            nose.tools.assert_almost_equal(v_ours, v_true, places = 5)

        for i in xrange(8):
            assert_routine_ok(
                numpy.random.normal(),
                numpy.random.normal(),
                numpy.random.rand() + 1.0,
                )

    def test_log_normal_log_pdf(self):
        def assert_routine_ok(x, mu, sigma, theta):
            v_ours = borg.statistics.log_normal_log_pdf(mu, sigma, theta, x)
            v_true = scipy.stats.lognorm.logpdf(x, sigma, loc = theta, scale = numpy.exp(mu))

            nose.tools.assert_almost_equal(v_ours, v_true)

        for i in xrange(8):
            mu = numpy.random.rand() * 10
            sigma = numpy.random.rand() * 10 + 1e-2
            theta = numpy.random.rand() * 10
            x = scipy.stats.lognorm.rvs(sigma, loc = theta, scale = numpy.exp(mu))

            assert_routine_ok(x, mu, sigma, theta)

    def test_log_normal_log_cdf(self):
        def assert_routine_ok(x, mu, sigma, theta):
            v_ours = borg.statistics.log_normal_log_cdf(mu, sigma, theta, x)
            v_true = scipy.stats.lognorm.logcdf(x, sigma, loc = theta, scale = numpy.exp(mu))

            nose.tools.assert_almost_equal(v_ours, v_true)

        assert_routine_ok(1.2, 0.0, 1.0, 0.0)
        assert_routine_ok(6000.0, 2.17, 0.28, 0.0)

        for i in xrange(8):
            mu = numpy.random.rand() * 10
            sigma = numpy.random.rand() * 10 + 1e-2
            theta = numpy.random.rand() * 10
            x = scipy.stats.lognorm.rvs(sigma, loc = theta, scale = numpy.exp(mu))

            assert_routine_ok(x, mu, sigma, theta)

    def test_binomial_log_pmf(self):
        def assert_routine_ok(p, N, n):
            v_ours = borg.statistics.binomial_log_pmf(p, N, n)
            v_true = scipy.stats.binom.logpmf(n, N, p)

            nose.tools.assert_almost_equal(v_ours, v_true)

        for i in xrange(8):
            assert_routine_ok(numpy.random.rand(), 10, numpy.random.randint(10))

    def test_multinomial_log_pmf(self):
        def assert_routine_ok(p, N, n):
            v_ours = \
                borg.statistics.multinomial_log_pmf(
                    numpy.array([p, 1.0 - p], dtype = numpy.double),
                    numpy.array([n, N - n], dtype = numpy.intc),
                    )
            v_true = scipy.stats.binom.logpmf(n, N, p)

            nose.tools.assert_almost_equal(v_ours, v_true)

        for i in xrange(8):
            assert_routine_ok(numpy.random.rand(), 10, numpy.random.randint(10))

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

