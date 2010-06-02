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

    from borg.portfolio.test.support import FakeAction
    from borg.portfolio.models       import FixedModel

    actions     = [FakeAction(i) for i in xrange(4)]
    predictions = dict((a, numpy.array([0.25, 0.125, 0.125, 0.50])) for a in actions)
    model       = FixedModel(predictions)

    assert_equal(model.predict([], None), predictions)

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

    random = RandomState(42)
    maps   = [model.predict([], random) for i in xrange(1024)]

    # verify that they're valid probabilities
    import numpy

    for predictions in maps:
        for (_, p) in predictions.items():
            assert_almost_equal(numpy.sum(p), 1.0)

    # verify that they're useless
    expected = numpy.ones(4) * 0.25

    for action in actions:
        mean  = sum(m[action] for m in maps)
        mean /= len(maps)

        assert_true(numpy.sum(numpy.abs(mean - expected)) < 0.05)

