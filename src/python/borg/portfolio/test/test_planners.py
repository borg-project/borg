"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_hard_myopic_planner():
    """
    Test the hard-maximizing myopic planner.
    """

    # build the model
    import numpy

    from borg.portfolio.test.support import FakeAction
    from borg.portfolio.models       import FixedModel

    actions     = [FakeAction("foo"), FakeAction("bar")]
    predictions = [
        [0.25, 0.00, 0.50, 0.25],
        [0.10, 0.10, 0.40, 0.40],
        ]
    model   = FixedModel(actions, numpy.array(predictions))

    # build and call the planner
    from borg.portfolio.planners import HardMyopicPlanner

    planner = HardMyopicPlanner(0.9)
    history = numpy.zeros((2, 4), numpy.uint)
    chosen  = planner.select(model, history, 32.0, None)

    # did it behave correctly?
    from nose.tools import assert_equal

    assert_equal(chosen.value, "bar")

def test_bellman_planner():
    """
    Test the Bellman replanning planner.
    """

    # build and call the planner
    import numpy

    from borg.portfolio.test.test_bellman import build_real_model
    from borg.portfolio.planners          import BellmanPlanner

    model   = build_real_model()
    planner = BellmanPlanner(2, 0.9)
    history = numpy.zeros((4, 2), numpy.uint)
    chosen  = planner.select(model, history, 32.0, None)

    # did it behave correctly?
    from nose.tools import assert_equal

    assert_equal(chosen.solver.name, "bar")

