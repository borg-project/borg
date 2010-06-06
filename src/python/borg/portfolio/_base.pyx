"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

cdef class Action:
    """
    An action in the world.
    """

    def __init__(self, double cost):
        """
        Initialize.
        """

        self.cost = cost

    def description(self):
        """
        A human-readable description of this action.
        """

        raise NotImplementedError()

    def outcomes(self):
        """
        The possible outcomes of this action.
        """

        raise NotImplementedError()

cdef class Outcome:
    """
    An outcome of an action in the world.
    """

    def __init__(self, double utility):
        """
        Initialize.
        """

        self.utility = utility

