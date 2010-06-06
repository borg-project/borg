"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from borg.portfolio.models import AbstractModel

class BellmanTestModel(AbstractModel):
    """
    Simple model for Bellman testing.
    """

    def __init__(self):
        """
        Initialize.
        """

        from borg.portfolio.test.support import (
            FakeAction,
            FakeOutcome,
            )

        outcomes      = map(FakeOutcome, [0.0, 1.0])
        self._actions = [FakeAction("foo", outcomes), FakeAction("bar", outcomes)]

    def predict(self, history, random):
        """
        Return the fixed map.
        """

        import numpy

        (foo_action, bar_action) = self._actions

        if numpy.sum(history[0]) >= numpy.sum(history[1]):
            return numpy.array([[0.25, 0.75], [0.75, 0.25]])
        else:
            return numpy.array([[0.75, 0.25], [0.25, 0.75]])

    @property
    def actions(self):
        """
        The actions associated with this model.
        """

        return self._actions

def test_compute_bellman():
    """
    Test Bellman-optimal planning.
    """

    # set up the world
    import numpy

    model         = BellmanTestModel()
    blank_history = numpy.zeros((2, 2), numpy.uint)

    # test the computation
    from nose.tools             import assert_equal
    from borg.portfolio.bellman import (
        compute_bellman_plan,
        compute_bellman_utility,
        )

    (expected_utility, best_plan) = compute_bellman_utility(model, 4, 128.0, 1.0, blank_history)

    assert_equal(expected_utility, 0.99609375)
    assert_equal(best_plan, [model.actions[0]] * 4)

    best_plan_again = compute_bellman_plan(model, 4, 128.0, 1.0)

    assert_equal(best_plan_again, [model.actions[0]] * 4)

