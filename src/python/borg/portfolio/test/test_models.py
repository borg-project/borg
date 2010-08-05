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

def test_distribution_model():
    """
    Test the probability-distribution portfolio model.
    """

    # set up the model
    from cargo.statistics.dcm        import DirichletCompoundMultinomial
    from cargo.statistics.tuple      import TupleDistribution
    from cargo.statistics.mixture    import FiniteMixture
    from borg.portfolio.models       import DistributionModel
    from borg.portfolio.test.support import (
        FakeAction,
        FakeOutcome,
        )

    actions = [
        FakeAction("foo", map(FakeOutcome, [0.0, 1.0])),
        FakeAction("bar", map(FakeOutcome, [0.0, 1.0])),
        ]
    mixture = \
        FiniteMixture(
            [0.25, 0.75],
            [
                TupleDistribution((
                    DirichletCompoundMultinomial([1e-4, 1.0],  1),
                    DirichletCompoundMultinomial([1.0,  1e-4], 1),
                    )),
                TupleDistribution((
                    DirichletCompoundMultinomial([1.0,  1e-4], 1),
                    DirichletCompoundMultinomial([1e-4, 1.0],  1),
                    )),
                ],
            )
    model   = DistributionModel(mixture, actions)

    # verify its a priori predictions
    import numpy

    from numpy.random  import RandomState
    from cargo.testing import assert_almost_equal_deep

    rs = RandomState(42)

    assert_almost_equal_deep(
        model.predict(numpy.zeros((2, 2), numpy.uint), rs).tolist(),
        [
            [0.74995000499950015, 0.25004999500049996],
            [0.25004999500049996, 0.74995000499950015],
            ],
        )

    # verify its a posteriori predictions
    assert_almost_equal_deep(
        model.predict(numpy.array([[1, 0], [0, 1]], numpy.uint), rs).tolist(),
        [
            [0.99990000666633339, 0.00010001999499990015],
            [0.00010001999499990015, 0.99990000666633339],
            ],
        )
    assert_almost_equal_deep(
        model.predict(numpy.array([[0, 1], [1, 0]], numpy.uint), rs).tolist(),
        [
            [0.00010001999499990015, 0.99990000666633339],
            [0.99990000666633339, 0.00010001999499990015],
            ],
        )

