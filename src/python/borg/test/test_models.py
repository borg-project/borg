"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import nose.tools
import borg

def test_fit_dcm_minka(outcomes, weights):
    """Test DCM fitting via Minka's fixed-point iteration."""

    alpha = [0.5, 2.5, 1.0]
    betas = numpy.random.dirichlet(alpha, size = 1024)
    counts = [numpy.random.multinomial(10, beta) for beta in betas]
    fitted = fit_dcm_minka(counts, numpy.ones(1024))

    print alpha
    print fitted

    assert numpy.all(alpha == fitted)

