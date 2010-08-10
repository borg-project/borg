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

        self._cost = cost

    def __reduce__(self):
        """
        Reduce this instance for pickling.
        """

        raise NotImplementedError()

    def description(self):
        """
        A human-readable name for this action.
        """

        raise NotImplementedError()

    def outcomes(self):
        """
        The possible outcomes of this action.
        """

        raise NotImplementedError()

    @property
    def cost(self):
        """
        Return the cost of this action.
        """

        return self._cost

cdef class Outcome:
    """
    An outcome of an action in the world.
    """

    def __init__(self, double utility):
        """
        Initialize.
        """

        self._utility = utility

    def __reduce__(self):
        """
        Reduce this instance for pickling.
        """

        return (Outcome, (self._utility,))

    @property
    def utility(self):
        """
        Return the utility of this outcome.
        """

        return self._utility

