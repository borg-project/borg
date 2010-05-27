"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools                  import assert_equal
from borg.portfolio.test.support import FakeAction

def get_action(strategy, task, budget, outcome):
    """
    Return a strategy's next action.
    """

    import numpy

    action_generator = strategy.select(task, budget, numpy.random)
    action           = action_generator.send(None)

    try:
        action_generator.send(outcome)
    except StopIteration:
        pass

    return action

def test_sequence_strategy():
    """
    Test the sequence selection strategy.
    """

    from itertools                 import cycle
    from functools                 import partial
    from borg.portfolio.strategies import SequenceStrategy

    actions  = [FakeAction(i) for i in xrange(4)]
    strategy = SequenceStrategy(cycle(actions))
    getter   = partial(get_action, strategy, None, 128.0, None)

    # verify basic behavior
    for action in actions:
        assert_equal(getter().value, action.value)

    # verify repeated behavior
    for action in actions:
        assert_equal(getter().value, action.value)

    # verify budget awareness
    selected = get_action(strategy, None, 2.0, None)

    assert_equal(selected, None)

def test_fixed_strategy():
    """
    Test the fixed selection strategy.
    """

    from functools                 import partial
    from borg.portfolio.strategies import FixedStrategy

    strategy = FixedStrategy(FakeAction(42))
    getter   = partial(get_action, strategy, None, 128.0, None)

    for i in xrange(4):
        assert_equal(getter().value, 42)

def test_modeling_selection_strategy():
    """
    Test the modeling selection strategy.
    """

    import numpy

    from borg.portfolio.models     import FixedModel
    from borg.portfolio.planners   import HardMyopicPlanner
    from borg.portfolio.strategies import ModelingStrategy

    # set up the strategy
    actions    = [FakeAction(i) for i in xrange(4)]
    prediction = dict(zip(actions, numpy.eye(4)))
    model      = FixedModel(prediction)
    planner    = HardMyopicPlanner(1.0)
    strategy   = ModelingStrategy(model, planner)

    # does it select the expected action?
    selected = get_action(strategy, None, 42.0, None)

    assert_equal(selected, actions[-1])

    # does it pay attention to feasibility?
    selected = get_action(strategy, None, 1.0, None)

    assert_equal(selected, None)

