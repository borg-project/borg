"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def yield_sanitized_opb(source):
    """
    Filter an OPB file to make picky solvers happy.
    """

    for line in source:
        yield line

