"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import nose.tools
import borg

def test_sampled_pmfs_log_pmf():
    """Test log-PMF computation over sampled PMFs."""

    cdfs = \
        numpy.array([
            [[0.1, 0.9], [0.9, 0.1]],
            [[0.1, 0.9], [0.9, 0.1]],
            ])
    counts = \
        numpy.array([
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
            [[1, 0], [2, 0]],
            ])
    logs = borg.models.sampled_pmfs_log_pmf(cdfs, counts)

    nose.tools.assert_almost_equal(numpy.exp(logs[0]), 0.1)
    nose.tools.assert_almost_equal(numpy.exp(logs[1]), 0.9**2)
    nose.tools.assert_almost_equal(numpy.exp(logs[2]), 0.1 * 0.9**2)

def test_kernel_model_sample():
    """Test sampling from a KDE model."""

    successes = numpy.array([[0, 1], [1, 0], [0, 0]], numpy.intc)
    failures = numpy.array([[0, 0], [0, 0], [0, 1]], numpy.intc)
    durations = \
        numpy.array([
            [[numpy.nan], [42.0]],
            [[24.0], [numpy.nan]],
            [[numpy.nan], [numpy.nan]],
            ])
    kernel = borg.models.DeltaKernel()
    model = borg.models.KernelModel(successes, failures, durations, 100.0, 1.0 + 1e-8, kernel)
    samples = model.sample(10, 4)

    nose.tools.assert_equal(samples.shape[0], 10)
    nose.tools.assert_true(numpy.all(numpy.logaddexp.reduce(samples, axis = -1) < 1e-10))

