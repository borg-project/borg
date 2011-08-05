"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

cdef class Kernel(object):
    cdef double integrate(self, double x)

cdef class Posterior(object):
    pass

