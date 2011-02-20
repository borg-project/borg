"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_sequence_strategy():
    """
    Test the sequence selection strategy.
    """

    from nose.tools                  import assert_equal
    from borg.portfolio.test.support import FakeAction
    from borg.portfolio.strategies   import SequenceStrategy

    actions  = [FakeAction(i) for i in xrange(4)]
    strategy = SequenceStrategy(actions * 2)

    # verify basic behavior
    strategy.reset()

    for action in actions:
        assert_equal(strategy.choose(128.0, None).value, action.value)

    assert_equal(strategy.choose(128.0, None).value, actions[0].value)

    # verify repeated behavior
    strategy.reset()

    for action in actions:
        assert_equal(strategy.choose(128.0, None).value, action.value)

    # verify budget awareness
    strategy.reset()

    assert_equal(strategy.choose(2.0, None), None)

def test_fixed_strategy():
    """
    Test the fixed selection strategy.
    """

    from nose.tools                  import assert_equal
    from borg.portfolio.test.support import FakeAction
    from borg.portfolio.strategies   import FixedStrategy

    strategy = FixedStrategy(FakeAction(42))

    # verify basic behavior
    strategy.reset()

    for _ in xrange(4):
        assert_equal(strategy.choose(128.0, None).value, 42)

    # verify budget awareness
    assert_equal(strategy.choose(2.0, None), None)

def test_modeling_selection_strategy():
    """
    Test the modeling selection strategy.
    """

    import numpy

    from nose.tools                  import assert_equal
    from borg.portfolio.test.support import FakeAction
    from borg.portfolio.models       import FixedModel
    from borg.portfolio.planners     import HardMyopicPlanner
    from borg.portfolio.strategies   import ModelingStrategy

    # set up the strategy
    actions    = [FakeAction(i) for i in xrange(4)]
    model      = FixedModel(actions, numpy.eye(4))
    planner    = HardMyopicPlanner(1.0)
    strategy   = ModelingStrategy(model, planner)

    # does it select the expected action?
    strategy.reset()

    assert_equal(strategy.choose(128.0, None), actions[-1])

    # does it pay attention to feasibility?
    assert_equal(strategy.choose(2.0, None), None)

