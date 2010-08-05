"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_compute_features():
    """
    Test computation of SAT instance features.
    """

    from os.path      import (
        join,
        dirname,
        )
    from nose.tools   import assert_equal
    from borg.sat.cnf import compute_features

    features = compute_features(join(dirname(__file__), "s57-100.cnf"))

    assert_equal(len(features), 48)

