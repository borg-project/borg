"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    )

def test_fixed_action_model():
    """
    Test the fixed-action portfolio model.
    """

    import numpy

    from utexas.portfolio.test.test_strategies import FakeAction
    from utexas.portfolio.models               import FixedActionModel

    actions     = [FakeAction(i) for i in xrange(4)]
    predictions = dict((a, numpy.array([0.25, 0.125, 0.125, 0.50])) for a in actions)
    model       = FixedActionModel(predictions)

    assert_equal(model.predict(None, []), predictions)

def test_random_action_model():
    """
    Test the random-action portfolio model.
    """

    # set up the model
    from utexas.portfolio.test.support import FakeAction
    from utexas.portfolio.models       import RandomActionModel

    actions = [FakeAction(i) for i in xrange(4)]
    model   = RandomActionModel(actions)

    # generate a bunch of predictions
    maps = [model.predict(None, []) for i in xrange(1024)]

    # verify that they're valid probabilities
    import numpy

    for predictions in maps:
        for (_, p) in predictions.items():
            assert_almost_equal(numpy.sum(p), 1.0)

    # verify that they're useless
    mean = numpy.ones(4) * 0.25

    for action in actions:
        total  = sum(m[action] for m in maps)
        total /= len(maps)

        assert_true(numpy.sum(numpy.abs(total - mean)) < 0.05)

