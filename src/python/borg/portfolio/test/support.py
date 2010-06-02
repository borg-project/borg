"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from borg.portfolio.world import (
    AbstractAction,
    AbstractOutcome,
    )

class FakeAction(AbstractAction):
    """
    An action strictly for testing.
    """

    def __init__(self, value, outcomes = None):
        """
        Initialize.
        """

        self.value = value

        if outcomes is None:
            self._outcomes = map(FakeOutcome, [0.0, 1.0, 2.0, 3.0])
        else:
            self._outcomes = outcomes

    @property
    def cost(self):
        """
        An arbitrary fixed cost.
        """

        return 16.0

    @property
    def outcomes(self):
        """
        Possible outcomes of this action.
        """

        return self._outcomes

    @property
    def description(self):
        """
        Describe this action.
        """

        return "%s" % self.value

class FakeOutcome(AbstractOutcome):
    """
    An outcome strictly for testing.
    """

    def __init__(self, utility):
        """
        Initialize.
        """

        self._utility = utility

    @property
    def utility(self):
        """
        The utility of this outcome.
        """

        return self._utility

