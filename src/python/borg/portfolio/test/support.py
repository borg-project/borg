"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from borg.portfolio.world import (
    Action,
    Outcome,
    )

class FakeAction(Action):
    """
    An action strictly for testing.
    """

    def __init__(self, value, outcomes = None):
        """
        Initialize.
        """

        Action.__init__(self, 16.0)

        self.value = value

        if outcomes is None:
            self._outcomes = map(FakeOutcome, [0.0, 1.0, 2.0, 3.0])
        else:
            self._outcomes = outcomes

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

class FakeOutcome(Outcome):
    """
    An outcome strictly for testing.
    """

    def __init__(self, utility):
        """
        Initialize.
        """

        Outcome.__init__(self, utility)

