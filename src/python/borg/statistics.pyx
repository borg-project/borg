"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import scipy.special

def inverse_digamma(x):
    """Return the (approximate) inverse of the digamma function."""

    if x >= -2.22:
        y0 = numpy.exp(x) + 0.5
    else:
        y0 = -1.0 / (x - scipy.special.digamma(1.0))

    f = lambda y: scipy.special.digamma(y) - x
    f_ = lambda y: scipy.special.polygamma(1, y)

    return scipy.optimize.newton(f, y0, fprime = f_)

def fit_dirichlet(vectors, weights):
    """Compute the maximum-likelihood Dirichlet distribution."""

    log_pbar_k = numpy.sum(weights[:, None, None] * numpy.log(vectors), axis = 0) / numpy.sum(weights)
    alpha = numpy.random.random(vectors.shape[1:])
    alpha /= numpy.sum(alpha, axis = 1)[:, None]
    last_alpha = alpha

    for i in xrange(64):
        psi_total = scipy.special.digamma(numpy.sum(alpha, axis = 1))
        psi_alpha = psi_total[:, None] + log_pbar_k

        alpha = numpy.array([[inverse_digamma(x) for x in row.flatten()] for row in psi_alpha])
        alpha = alpha.reshape(psi_alpha.shape)

        if numpy.sum(numpy.abs(alpha - last_alpha)) <= 1e-10:
            break

        last_alpha = alpha

    return alpha

