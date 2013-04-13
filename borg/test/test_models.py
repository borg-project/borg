"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import nose.tools
import borg

def test_sampled_pmfs_log_pmf():
    """Test borg.models.sampled_pmfs_log_pmf()."""

    cdfs = \
        numpy.log([
            [[0.1, 0.9], [0.9, 0.1]],
            [[0.1, 0.9], [0.9, 0.1]],
            ])
    counts = \
        numpy.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [2, 0]],
                [[1, 0], [2, 0]],
                ],
            numpy.intc,
            )
    logs = borg.models.sampled_pmfs_log_pmf(cdfs, counts)

    nose.tools.assert_almost_equal(numpy.exp(logs[0, 0]), 0.1)
    nose.tools.assert_almost_equal(numpy.exp(logs[0, 1]), 0.9**2)
    nose.tools.assert_almost_equal(numpy.exp(logs[0, 2]), 0.1 * 0.9**2)

def test_kernel_model_sample():
    """Test borg.models.KernelModel.sample()."""

    successes = numpy.array([[0, 1], [1, 0], [0, 0]], numpy.intc)
    failures = numpy.array([[0, 0], [0, 0], [0, 1]], numpy.intc)
    durations = \
        numpy.array([
            [[numpy.nan], [42.0]],
            [[24.0], [numpy.nan]],
            [[numpy.nan], [numpy.nan]],
            ])
    kernel = borg.models.DeltaKernel()
    alpha = 1.0 + 1e-8
    model = borg.models.KernelModel(successes, failures, durations, 100.0, alpha, kernel)
    samples = model.sample(16, 4)

    nose.tools.assert_true(numpy.all(numpy.logaddexp.reduce(samples, axis = -1) < 1e-10))
    nose.tools.assert_true(numpy.any(numpy.abs(samples[..., 0] - numpy.log((alpha) / (5 * alpha - 4))) < 1e-10))
    nose.tools.assert_true(numpy.any(numpy.abs(samples[..., -1] - numpy.log(1.0 / 5)) < 1e-10))
    nose.tools.assert_true(numpy.any(numpy.abs(samples[..., -1] - numpy.log((alpha) / (5 * alpha - 4))) < 1e-10))
    nose.tools.assert_true(numpy.any(numpy.abs(samples[..., -1] - numpy.log((alpha - 1) / (5 * alpha - 4))) < 1e-10))

def test_multinomial_model_fit():
    """Test borg.models.MultinomialModel.fit()."""

    runs = [
        ("solver_a", 100.0, 1.0, True),
        ("solver_a", 100.0, 48.0, True),
        ("solver_a", 100.0, 100.0, False),
        ("solver_b", 100.0, 66.0, True),
        ("solver_b", 100.0, 77.0, True),
        ("solver_b", 100.0, 100.0, False),
        ]

    training = borg.RunData()

    for run in runs:
        training.add_run("foo", borg.storage.RunRecord(*run))
        training.add_run("bar", borg.storage.RunRecord(*run))

    alpha = 1.0 + 1e-8
    model = borg.models.MultinomialModel.fit(["solver_a", "solver_b"], training, 4, alpha)
    components = numpy.exp(model.log_components)

    nose.tools.assert_true(numpy.all(numpy.abs(components[0] - components[1]) == 0.0))
    nose.tools.assert_true(numpy.all(numpy.abs(numpy.sum(components, axis = -1) - 1.0) < 1e-10))
    nose.tools.assert_true(numpy.all(components[:, 0, 0] == components[:, 0, 1]))
    nose.tools.assert_true(numpy.all(components[:, 0, 1] > components[:, 0, 2]))
    nose.tools.assert_true(numpy.all(components[:, 1, 0] == components[:, 1, 1]))
    nose.tools.assert_true(numpy.all(components[:, 1, 1] < components[:, 1, 2]))

def test_multinomial_model_condition():
    model = borg.models.MultinomialModel(10.0, numpy.log([[[0.2, 0.1]], [[0.9, 0.8]]]), numpy.log([0.5, 0.5]))
    posterior0 = model.condition([(0, 0)])
    posterior1 = model.condition([(0, 1)])

    nose.tools.assert_almost_equal(posterior0.log_weights[0], numpy.log(0.2 * 0.5 / (0.2 * 0.5 + 0.9 * 0.5)))
    nose.tools.assert_almost_equal(posterior0.log_weights[1], numpy.log(0.9 * 0.5 / (0.2 * 0.5 + 0.9 * 0.5)))
    nose.tools.assert_almost_equal(posterior1.log_weights[0], numpy.log(0.1 * 0.5 / (0.1 * 0.5 + 0.8 * 0.5)))
    nose.tools.assert_almost_equal(posterior1.log_weights[1], numpy.log(0.8 * 0.5 / (0.1 * 0.5 + 0.8 * 0.5)))

