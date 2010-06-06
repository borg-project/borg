"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

cdef class AbstractAction:
    cdef double cost

cdef class AbstractOutcome:
    cdef double utility

