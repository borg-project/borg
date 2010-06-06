"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

cdef class Predictor:
    """
    Core of a performance-tuned model.
    """

    cdef int predict_raw(self, unsigned int* history, double* out)

