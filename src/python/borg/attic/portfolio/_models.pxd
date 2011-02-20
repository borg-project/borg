"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

cdef class Predictor:
    """
    Core of a performance-tuned model.
    """

    cdef int predict_raw(self, unsigned int* history, double* out)

cdef class SlowPredictor(Predictor):
    """
    Provide an interface to slow Python-implemented prediction routines.
    """

    cdef object _predict
    cdef size_t _M
    cdef size_t _D

