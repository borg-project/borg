"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

cdef class Kernel(object):
    cdef double log_density(self, double x, double v)
    cdef double integrate(self, double x, double v)

cdef class Posterior(object):
    pass

