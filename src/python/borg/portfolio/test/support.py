"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.portfolio.world import (
    Action,
    Outcome,
    )

class FakeAction(Action):
    """
    An action strictly for testing.
    """

    def __init__(self, value):
        """
        Initialize.
        """

        self.value     = value
        self._outcomes = map(FakeOutcome, [0.0, 1.0, 2.0, 3.0])

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

class FakeOutcome(Outcome):
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

