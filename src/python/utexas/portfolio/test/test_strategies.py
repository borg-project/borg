"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools             import assert_equal
from utexas.portfolio.world import Action

class FakeAction(Action):
    """
    An action strictly for testing.
    """

    def __init__(self, value):
        """
        Initialize.
        """

        self.value = value

    @property
    def cost(self):
        """
        An arbitrary fixed cost.
        """

        return 16.0

def get_action(strategy, task, budget, outcome):
    """
    Return a strategy's next action.
    """

    action_generator = strategy.select(task, budget)
    action           = action_generator.send(None)

    try:
        action_generator.send(outcome)
    except StopIteration:
        pass

    return action

def test_sequence_selection_strategy():
    """
    Test the sequence selection strategy.
    """

    from itertools                   import cycle
    from functools                   import partial
    from utexas.portfolio.strategies import SequenceSelectionStrategy

    actions  = [FakeAction(i) for i in xrange(4)]
    strategy = SequenceSelectionStrategy(cycle(actions))
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

def test_fixed_selection_strategy():
    """
    Test the fixed selection strategy.
    """

    from functools                   import partial
    from utexas.portfolio.strategies import FixedSelectionStrategy

    strategy = FixedSelectionStrategy(FakeAction(42))
    getter   = partial(get_action, strategy, None, 128.0, None)

    for i in xrange(4):
        assert_equal(getter().value, 42)

