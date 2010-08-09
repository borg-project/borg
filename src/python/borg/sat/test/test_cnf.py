"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_compute_features():
    """
    Test computation of SAT instance features.
    """

    from nose.tools   import assert_equal
    from borg         import get_support_path
    from borg.sat.cnf import compute_features

    features = compute_features(get_support_path("s57-100.cnf"))

    assert_equal(len(features), 48)

