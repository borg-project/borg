"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import nose
import borg

def test_klmeans_simple():
    points = [[[0.90, 0.10]] * 2, [[0.91, 0.09]] * 2, [[0.11, 0.89]] * 2, [[0.09, 0.91]] * 2]
    kl_means = borg.bregman.KLMeans(k = 2).fit(points)

    (center0, center1) = kl_means.cluster_centers_

    if center1[0, 0] < center0[0, 0]:
        (center0, center1) = (center1, center0)

    nose.tools.assert_equal(center0.tolist(), [[0.100, 0.900]] * 2)
    nose.tools.assert_equal(center1.tolist(), [[0.905, 0.095]] * 2)

