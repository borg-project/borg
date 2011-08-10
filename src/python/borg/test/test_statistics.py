"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import scipy.special
import nose.tools
import borg

def test_digamma():
    """Test borg.statistics.digamma()."""

    # XXX move to the cephes implementation?
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e-2), scipy.special.digamma(1e-2), places = 2)
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e-1), scipy.special.digamma(1e-1), places = 2)
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e-0), scipy.special.digamma(1e-0), places = 2)
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e+1), scipy.special.digamma(1e+1), places = 2)
    nose.tools.assert_almost_equal(borg.statistics.digamma(1e+2), scipy.special.digamma(1e+2), places = 2)

def test_dcm_fit():
    """Test borg.statistics.dcm_estimate_ml()."""

    counts = numpy.array([[4, 4], [4, 4]], numpy.intc)
    alpha = borg.statistics.dcm_estimate_ml(counts)
    mean = alpha / numpy.sum(alpha)

    nose.tools.assert_almost_equal(mean[0], mean[1])

