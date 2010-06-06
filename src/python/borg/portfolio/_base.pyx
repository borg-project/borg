"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

cdef class AbstractAction:
    """
    An action in the world.
    """

    def __cinit__(self, double cost):
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

cdef class AbstractOutcome:
    """
    An outcome of an action in the world.
    """

