"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import scipy.stats
import scipy.special
import nose.tools
import borg

def test_unit_uniform_rv():
    def assert_ok():
        samples = numpy.array([borg.statistics.unit_uniform_rv() for _ in xrange(65535)])

        nose.tools.assert_true(numpy.all(samples >= 0.0))
        nose.tools.assert_true(numpy.all(samples < 1.0))
        nose.tools.assert_almost_equal(numpy.mean(samples), 0.5, places = 2)

    for _ in xrange(8):
        #borg.statistics.set_prng_seeds(numpy.random.randint(-sys.maxint - 1, sys.maxint))

        yield (assert_ok,)

def test_unit_normal_rv():
    samples = [borg.statistics.unit_normal_rv() for _ in xrange(65535)]

    nose.tools.assert_almost_equal(numpy.mean(samples), 0.0, places = 2)
    nose.tools.assert_almost_equal(numpy.std(samples), 1.0, places = 2)

def test_unit_gamma_rv():
    def assert_unit_gamma_rv_ok(shape):
        rv_samples = [borg.statistics.unit_gamma_rv(shape) for _ in xrange(65535)]
        np_samples = numpy.random.gamma(shape, size = 65535)

        nose.tools.assert_almost_equal(numpy.mean(rv_samples), numpy.mean(np_samples), places = 1)
        nose.tools.assert_almost_equal(numpy.std(rv_samples), numpy.std(np_samples), places = 1)

    yield (assert_unit_gamma_rv_ok, 1e-2)
    yield (assert_unit_gamma_rv_ok, 1e-1)
    yield (assert_unit_gamma_rv_ok, 1e-0)
    yield (assert_unit_gamma_rv_ok, 1e+1)
    yield (assert_unit_gamma_rv_ok, 1e+2)

def test_digamma():
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e-2), scipy.special.digamma(1e-2))
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e-1), scipy.special.digamma(1e-1))
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e-0), scipy.special.digamma(1e-0))
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e+1), scipy.special.digamma(1e+1))
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e+2), scipy.special.digamma(1e+2))

def test_inverse_digamma():
    def assert_inverse_digamma_ok(x):
        v = borg.statistics.digamma(x)

        nose.tools.assert_almost_equal(borg.statistics.inverse_digamma(v), x)

    yield (assert_inverse_digamma_ok, 1e-2)
    yield (assert_inverse_digamma_ok, 1e-1)
    yield (assert_inverse_digamma_ok, 1e-0)
    yield (assert_inverse_digamma_ok, 1e+1)
    yield (assert_inverse_digamma_ok, 1e+2)

def test_inverse_digamma_newton():
    def assert_inverse_digamma_ok(x):
        v = borg.statistics.digamma(x)

        nose.tools.assert_almost_equal(borg.statistics.inverse_digamma_newton(v), x)

    yield (assert_inverse_digamma_ok, 1e-2)
    yield (assert_inverse_digamma_ok, 1e-1)
    yield (assert_inverse_digamma_ok, 1e-0)
    yield (assert_inverse_digamma_ok, 1e+1)
    yield (assert_inverse_digamma_ok, 1e+2)

def test_gamma_log_pdf():
    def assert_ok(x):
        nose.tools.assert_almost_equal(
            borg.statistics.gamma_log_pdf(x, 1.1, 2.0),
            scipy.stats.gamma.logpdf(x, 1.1, scale = 2.0),
            )

    with borg.util.numpy_errors(all = "raise"):
        for i in xrange(4):
            x = numpy.random.rand() * 10

            yield (assert_ok, x)

def test_dirichlet_estimate_ml_simple():
    with borg.util.numpy_errors(all = "raise"):
        a = 1e-2
        b = 1.0 - 1e-2

        vectors = numpy.array([[a, b]] * 16, numpy.double)
        alpha = borg.statistics.dirichlet_estimate_ml(vectors)
        mean = alpha / numpy.sum(alpha)

        nose.tools.assert_true(alpha[0] < alpha[1])
        nose.tools.assert_true(numpy.all(alpha > 0.0))
        nose.tools.assert_almost_equal(mean[0], a, places = 1)
        nose.tools.assert_almost_equal(mean[1], b, places = 1)

def test_dirichlet_estimate_map_simple():
    with borg.util.numpy_errors(all = "raise"):
        a = 1e-2
        b = 1.0 - 1e-2

        vectors = numpy.array([[a, b]] * 16, numpy.double)
        alpha = borg.statistics.dirichlet_estimate_map(vectors, shape = 1.1, scale = 2.0)
        mean = alpha / numpy.sum(alpha)

        nose.tools.assert_true(alpha[0] < alpha[1])
        nose.tools.assert_true(numpy.all(alpha > 0.0))
        nose.tools.assert_almost_equal(mean[0], a, places = 1)
        nose.tools.assert_almost_equal(mean[1], b, places = 1)

def test_dirichlet_estimate_ml_zeros():
    with borg.util.numpy_errors(all = "raise"):
        a = 0.0
        b = 1.0

        vectors = numpy.array([[a, b]] * 16, numpy.double)
        alpha = borg.statistics.dirichlet_estimate_ml(vectors)
        mean = alpha / numpy.sum(alpha)

        nose.tools.assert_true(alpha[0] < alpha[1])
        nose.tools.assert_almost_equal(mean[0], a, places = 1)
        nose.tools.assert_almost_equal(mean[1], b, places = 1)

def test_dirichlet_estimate_map_zeros():
    with borg.util.numpy_errors(all = "raise"):
        a = 0.0
        b = 1.0

        vectors = numpy.array([[a, b]] * 16, numpy.double)
        alpha = borg.statistics.dirichlet_estimate_map(vectors)
        mean = alpha / numpy.sum(alpha)

        nose.tools.assert_true(alpha[0] < alpha[1])
        nose.tools.assert_almost_equal(mean[0], a, places = 1)
        nose.tools.assert_almost_equal(mean[1], b, places = 1)

def test_dirichlet_estimate_ml_disjoint():
    with borg.util.numpy_errors(all = "raise"):
        a = 0.0
        b = 1.0

        vectors = numpy.array([[a, b]] * 16 + [[b, a]] * 16, numpy.double)
        alpha = borg.statistics.dirichlet_estimate_ml(vectors)

        nose.tools.assert_almost_equal(alpha[0], alpha[1])

def test_dirichlet_estimate_map_disjoint():
    with borg.util.numpy_errors(all = "raise"):
        a = 0.0
        b = 1.0

        vectors = numpy.array([[a, b]] * 16 + [[b, a]] * 16, numpy.double)
        alpha = borg.statistics.dirichlet_estimate_map(vectors, shape = 30, scale = 2)

        nose.tools.assert_almost_equal(alpha[0], alpha[1])

def test_dirichlet_estimate_ml_random():
    def assert_ok(alpha):
        vectors = numpy.random.dirichlet(alpha, 65536)
        estimate = borg.statistics.dirichlet_estimate_ml(vectors)

        nose.tools.assert_almost_equal(estimate[0], alpha[0], places = 1)
        nose.tools.assert_almost_equal(estimate[1], alpha[1], places = 1)

    with borg.util.numpy_errors(all = "raise"):
        for i in xrange(8):
            alpha = numpy.random.dirichlet(numpy.ones(2)) + 1e-1

            yield (assert_ok, alpha.tolist())

def test_dirichlet_estimate_map_random():
    def assert_ok(alpha):
        vectors = numpy.random.dirichlet(alpha, 65536)
        estimate = borg.statistics.dirichlet_estimate_map(vectors)

        nose.tools.assert_almost_equal(estimate[0], alpha[0], places = 1)
        nose.tools.assert_almost_equal(estimate[1], alpha[1], places = 1)

    with borg.util.numpy_errors(all = "raise"):
        for i in xrange(8):
            alpha = numpy.random.dirichlet(numpy.ones(2)) + 1e-1

            yield (assert_ok, alpha.tolist())

def test_dcm_log_pdf():
    def assert_ok(vector, alpha):
        density = borg.statistics.dcm_pdf(alpha, vector)
        log_density = borg.statistics.dcm_log_pdf(alpha, vector)

        nose.tools.assert_almost_equal(log_density, numpy.log(density))

    with borg.util.numpy_errors(all = "raise"):
        for i in xrange(8):
            yield (
                assert_ok,
                numpy.random.randint(4, size = 4),
                numpy.random.rand(4),
                )

def test_dcm_estimate_ml_simple():
    counts = numpy.array([[4, 4], [4, 4]], numpy.intc)
    alpha = borg.statistics.dcm_estimate_ml(counts)

    nose.tools.assert_almost_equal(alpha[0], alpha[1])

def test_dcm_estimate_ml_random():
    def assert_ok(alpha):
        vectors = numpy.random.dirichlet(alpha, 65536)
        counts = numpy.empty_like(vectors)

        for n in xrange(vectors.shape[0]):
            counts[n] = numpy.random.multinomial(8, vectors[n])

        estimate = borg.statistics.dcm_estimate_ml(counts)

        nose.tools.assert_almost_equal(estimate[0], alpha[0], places = 1)
        nose.tools.assert_almost_equal(estimate[1], alpha[1], places = 1)

    with borg.util.numpy_errors(all = "raise"):
        for i in xrange(8):
            alpha = numpy.random.dirichlet(numpy.ones(2)) + numpy.random.rand() + 1e-1

            yield (assert_ok, alpha.tolist())

def test_dcm_estimate_ml_wallach_simple():
    counts = numpy.array([[4, 4], [4, 4]], numpy.intc)
    alpha = borg.statistics.dcm_estimate_ml_wallach(counts)

    nose.tools.assert_almost_equal(alpha[0], alpha[1])

def test_dcm_estimate_ml_wallach_random():
    def assert_ok(alpha):
        vectors = numpy.random.dirichlet(alpha, 65536)
        counts = numpy.empty_like(vectors).astype(numpy.intc)

        for n in xrange(vectors.shape[0]):
            counts[n] = numpy.random.multinomial(8, vectors[n])

        estimate = borg.statistics.dcm_estimate_ml_wallach(counts)

        nose.tools.assert_almost_equal(estimate[0], alpha[0], places = 1)
        nose.tools.assert_almost_equal(estimate[1], alpha[1], places = 1)

    with borg.util.numpy_errors(all = "raise"):
        for i in xrange(8):
            alpha = numpy.random.dirichlet(numpy.ones(2)) + numpy.random.rand() + 1e-1

            yield (assert_ok, alpha.tolist())

def test_log_normal_estimate_ml():
    def assert_ok(mu, sigma, theta, terminus):
        values = numpy.exp(numpy.random.normal(mu, sigma, 64000)) + theta
        uncensored = values[values < terminus]

        (e_mu, e_sigma, e_theta) = \
            borg.statistics.log_normal_estimate_ml(
                uncensored,
                numpy.zeros(uncensored.size, dtype = numpy.intc),
                numpy.array([1.0]),
                numpy.array([values.size - uncensored.size], dtype = numpy.intc),
                terminus,
                )

        nose.tools.assert_almost_equal(e_mu, mu, places = 2)
        nose.tools.assert_almost_equal(e_sigma, sigma, places = 2)
        nose.tools.assert_almost_equal(e_theta, theta, places = 2)

    yield (assert_ok, 10.0, 1.0, 0.0, 10.0)
    #yield (assert_ok, 0.0, 1.0, 0.0, 1.0)
    #yield (assert_ok, 0.0, 1.0, 10.0, 15.0)

def test_log_normal_mixture_estimate_ml():
    def assert_ok(mus, sigmas, thetas, terminus):
        instances_per_component = 2
        samples_per_instance = 6
        times = []
        ns = []
        censored = []
        n = 0

        for (mu, sigma, theta) in zip(mus, sigmas, thetas):
            for _ in xrange(instances_per_component):
                values = numpy.exp(numpy.random.normal(mu, sigma, samples_per_instance)) + theta
                uncensored = list(values[values <= terminus])
                times += uncensored
                censored += [numpy.sum(values > terminus)]
                ns += [n] * len(uncensored)
                n += 1

        print repr(times)
        print repr(ns)
        print repr(censored)

        borg.statistics.log_normal_mixture_estimate_ml(
            numpy.array(times),
            numpy.array(ns),
            numpy.array(censored),
            terminus,
            len(thetas),
            )

        # XXX actually assert something

    #with borg.util.numpy_errors(all = "raise"):
    yield (assert_ok, [0.0, 10.0], [1.0, 2.0], [0.0, 0.0], 1e4)

