"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    )

def test_fixed_model():
    """
    Test the fixed-prediction portfolio model.
    """

    import numpy

    from cargo.testing               import assert_almost_equal_deep
    from borg.portfolio.test.support import FakeAction
    from borg.portfolio.models       import FixedModel

    actions     = [FakeAction(i) for i in xrange(4)]
    predictions = numpy.array([[0.25, 0.125, 0.125, 0.50] for a in actions])
    model       = FixedModel(actions, predictions)

    assert_almost_equal_deep(model.predict([], None).tolist(), predictions.tolist())

def test_random_model():
    """
    Test the random-prediction portfolio model.
    """

    # set up the model
    from borg.portfolio.test.support import FakeAction
    from borg.portfolio.models       import RandomModel

    actions = [FakeAction(i) for i in xrange(4)]
    model   = RandomModel(actions)

    # generate a bunch of predictions
    from numpy.random import RandomState

    random    = RandomState(42)
    predicted = [model.predict([], random) for i in xrange(1024)]

    # verify that they're valid probabilities
    import numpy

    for predictions in predicted:
        for p in predictions:
            assert_almost_equal(numpy.sum(p), 1.0)

    # verify that they're useless
    expected = numpy.ones(4) * 0.25

    for (i, action) in enumerate(actions):
        mean  = sum(m[i] for m in predicted)
        mean /= len(predicted)

        assert_true(numpy.sum(numpy.abs(mean - expected)) < 0.05)

class FakeDCM_Estimator(object):
    """
    Estimate some fixed mixture.
    """

    def __init__(self, mixture):
        """
        Initialize.
        """

        self._mixture = mixture

    def estimate(self, samples):
        """
        Pretend to estimate.
        """

        return self._mixture

def test_dcm_model():
    """
    Test the DCM-mixture portfolio model.
    """

    # set up the model
    from cargo.statistics.dcm        import DirichletCompoundMultinomial
    from cargo.statistics.mixture    import FiniteMixture
    from borg.portfolio.models       import DCM_MixtureModel
    from borg.portfolio.test.support import (
        FakeAction,
        FakeOutcome,
        )

    foo_action = FakeAction("foo", map(FakeOutcome, [0.0, 1.0]))
    bar_action = FakeAction("bar", map(FakeOutcome, [0.0, 1.0]))
    mixture    = \
        FiniteMixture(
            [0.25, 0.75],
            [
                [
                    DirichletCompoundMultinomial([1e-4, 1.0]),
                    DirichletCompoundMultinomial([1.0,  1e-4]),
                    ],
                [
                    DirichletCompoundMultinomial([1.0,  1e-4]),
                    DirichletCompoundMultinomial([1e-4, 1.0]),
                    ],
                ],
            )
    estimator = FakeDCM_Estimator(mixture)
    training  = {foo_action: [], bar_action: []}
    model     = DCM_MixtureModel(training, estimator)

    # verify its a priori predictions
    import numpy
    import numpy.random as r

    from cargo.testing import assert_almost_equal_deep

    assert_almost_equal_deep(
        model.predict(numpy.zeros((2, 2), numpy.uint), r).tolist(),
        [
            [0.74995000499950015, 0.25004999500049996],
            [0.25004999500049996, 0.74995000499950015],
            ],
        )

    # verify its a posteriori predictions
    assert_almost_equal_deep(
        model.predict(numpy.array([[1, 0], [0, 1]], numpy.uint), r).tolist(),
        [
            [0.99995000083345764, 4.9999166541667325e-05],
            [4.9999166541667325e-05, 0.99995000083345764],
            ],
        )
    assert_almost_equal_deep(
        model.predict(numpy.array([[0, 1], [1, 0]], numpy.uint), r).tolist(),
        [
            [5.0012497874656265e-05, 0.9999499875021246],
            [0.9999499875021246, 5.0012497874656265e-05],
            ],
        )

