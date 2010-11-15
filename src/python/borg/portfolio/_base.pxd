"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

cdef class Action:
    """
    An action in the world.
    """

    cdef double _cost

cdef class Outcome:
    """
    The outcome of an action in the world.
    """

    cdef double _utility

