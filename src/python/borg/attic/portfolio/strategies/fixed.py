"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

class FixedStrategy(object):
    """
    A strategy the follows an iterable sequence for every task.
    """

    def __init__(self, action):
        """
        Initialize.
        """

        self._action = action

    def reset(self):
        """
        Prepare to solve a new task.
        """

    def see(self, action, outcome):
        """
        Witness the outcome of an action.
        """

    def choose(self, budget, random):
        """
        Return the selected action.
        """

        return self._action

