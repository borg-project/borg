"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def yield_selected(strategy, budget):
    """
    Return a strategy's next action.
    """

    from itertools    import repeat
    from numpy.random import RandomState

    selected = strategy.select(budget, RandomState(42))
    result   = None

    for outcome in repeat(None):
        action = selected.send(result)

        if action is not None:
            budget -= action.cost

        result = (outcome, budget)

        yield action

def test_sequence_strategy():
    """
    Test the sequence selection strategy.
    """

    from nose.tools                  import assert_equal
    from borg.portfolio.test.support import FakeAction
    from borg.portfolio.strategies   import SequenceStrategy

    actions  = [FakeAction(i) for i in xrange(4)]
    strategy = SequenceStrategy(actions * 2)
    selected = yield_selected(strategy, 128.0)

    # verify basic behavior
    for (action, selected_action) in zip(actions, selected):
        assert_equal(selected_action.value, action.value)

    assert_equal(selected.next().value, actions[0].value)

    # verify repeated behavior
    selected = yield_selected(strategy, 128.0)

    for (action, selected_action) in zip(actions, selected):
        assert_equal(selected_action.value, action.value)

    # verify budget awareness
    assert_equal(yield_selected(strategy, 2.0).next(), None)

def test_fixed_strategy():
    """
    Test the fixed selection strategy.
    """

    from nose.tools                  import assert_equal
    from borg.portfolio.test.support import FakeAction
    from borg.portfolio.strategies   import FixedStrategy

    strategy = FixedStrategy(FakeAction(42))
    selected = yield_selected(strategy, 128.0)

    # verify basic behavior
    for (_, action) in zip(xrange(4), selected):
        assert_equal(action.value, 42)

    # verify budget awareness
    assert_equal(yield_selected(strategy, 2.0).next(), None)

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
    selected = yield_selected(strategy, 128.0).next()

    assert_equal(selected, actions[-1])

    # does it pay attention to feasibility?
    selected = yield_selected(strategy, 2.0).next()

    assert_equal(selected, None)

