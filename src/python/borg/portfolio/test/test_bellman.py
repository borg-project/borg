"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from borg.portfolio.models import AbstractModel

class BellmanTestModel(AbstractModel):
    """
    Simple model for Bellman testing.
    """

    def __init__(self, actions):
        """
        Initialize.
        """

        assert len(actions) == 2

        self._actions = actions

    def predict(self, history, random):
        """
        Return the fixed map.
        """

        (foo_action, bar_action) = self._actions

        counts = {foo_action: 0, bar_action: 0}

        for (a, o) in history:
            counts[a] += 1

        if counts[foo_action] >= counts[bar_action]:
            return {foo_action: [0.25, 0.75], bar_action: [0.75, 0.25]}
        else:
            return {foo_action: [0.75, 0.25], bar_action: [0.25, 0.75]}

def test_compute_bellman_utility():
    """
    Test computation of Bellman expected utility.
    """

    # set up the world
    from borg.portfolio.test.support import (
        FakeAction,
        FakeOutcome,
        )

    outcomes = map(FakeOutcome, [0.0, 1.0])
    actions  = [FakeAction("foo", outcomes), FakeAction("bar", outcomes)]
    model    = BellmanTestModel(actions)

    # test the computation
    from nose.tools             import assert_equal
    from borg.portfolio.bellman import compute_bellman_utility

    (expected_utility, best_plan) = compute_bellman_utility(model, 4, 128.0, [])

    assert_equal(expected_utility, 0.99609375)
    assert_equal(best_plan, [actions[0]] * 4)

