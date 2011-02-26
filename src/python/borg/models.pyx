"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import borg
import numpy
cimport numpy

def fit_binomial_mixture(observed, counts, K):
    """Use EM to fit a discrete mixture."""

    concentration = 1e-8
    N = observed.shape[0]
    rates = (observed + concentration / 2.0) / (counts + concentration)
    components = rates[numpy.random.randint(N, size = K)]
    responsibilities = numpy.empty((K, N))
    old_ll = -numpy.inf

    for i in xrange(512):
        # compute new responsibilities
        raw_mass = scipy.stats.binom.logpmf(observed[None, ...], counts[None, ...], components[:, None, ...])
        log_mass = numpy.sum(raw_mass, axis = 2)
        responsibilities = log_mass - numpy.logaddexp.reduce(log_mass, axis = 0)
        responsibilities = numpy.exp(responsibilities)
        weights = numpy.sum(responsibilities, axis = 1)
        weights /= numpy.sum(weights)
        ll = numpy.sum(numpy.logaddexp.reduce(numpy.log(weights)[:, None] + log_mass, axis = 0))

        logger.debug("l-l at iteration %i is %f", i, ll)

        # compute new components
        map_observed = observed[None, ...] * responsibilities[..., None]
        map_counts = counts[None, ...] * responsibilities[..., None]
        components = numpy.sum(map_observed, axis = 1) + concentration / 2.0
        components /= numpy.sum(map_counts, axis = 1) + concentration

        # check for termination
        if numpy.abs(ll - old_ll) <= 1e-3:
            break

        old_ll = ll

    assert numpy.all(components >= 0.0)
    assert numpy.all(components <= 1.0)

    return (components, weights, responsibilities, ll)

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
    log_pbar_k = numpy.sum(weights[:, None, None] * numpy.log(vectors), axis = 0) / numpy.sum(weights)
    alpha = numpy.random.random(vectors.shape[1:])
    alpha /= numpy.sum(alpha, axis = 1)[:, None]

    for i in xrange(64):
        psi_total = scipy.special.digamma(numpy.sum(alpha, axis = 1))
        psi_alpha = psi_total[:, None] + log_pbar_k
        alpha = numpy.array([[inverse_digamma(x) for x in row.flatten()] for row in psi_alpha])
        alpha = alpha.reshape(psi_alpha.shape)

        # XXX termination condition!

    return alpha

def fit_dirichlet_vfixed(vectors, weights, variance):
    log_pbar_k = numpy.sum(weights[:, None, None] * numpy.log(vectors), axis = 0) / numpy.sum(weights)
    alpha = numpy.random.random(vectors.shape[1:])
    alpha /= numpy.sum(alpha, axis = 1)[:, None]
    last_alpha = alpha

    for i in xrange(64):
        psi_full = scipy.special.digamma(variance * alpha)
        psi_sigma = numpy.sum(alpha * (log_pbar_k - psi_full), axis = -1)
        psi_alpha = log_pbar_k - psi_sigma[..., None]

        alpha = numpy.array([[inverse_digamma(x) for x in row.flatten()] for row in psi_alpha])
        alpha = alpha.reshape(psi_alpha.shape)
        alpha /= numpy.sum(alpha, axis = -1)[..., None]

        if numpy.sum(numpy.abs(alpha - last_alpha)) <= 1e-10:
            break

        last_alpha = alpha

    return alpha * variance

def dirichlet_log_pdf(vectors, alphas):
    """Compute the Dirichlet log PDF."""

    vectors = numpy.asarray(vectors)
    alphas = numpy.asarray(alphas)

    term_a = scipy.special.gammaln(numpy.sum(alphas, axis = -1))
    term_b = numpy.sum(scipy.special.gammaln(alphas), axis = -1)
    term_c = numpy.sum((alphas - 1.0) * numpy.log(vectors), axis = -1)

    return term_a - term_b + term_c

def dcm_pdf(vector, alpha):
    """Compute the Dirichlet log PDF."""

    sum_alpha = numpy.sum(alpha, axis = -1)
    sum_vector = numpy.sum(vector, axis = -1)

    term_l = scipy.special.gamma(sum_alpha) / scipy.special.gamma(sum_alpha + sum_vector)
    term_r = numpy.prod(scipy.special.gamma(vector + alpha) / scipy.special.gamma(alpha), axis = -1)

    return term_l * term_r

def fit_dirichlet_mixture(vectors, K, variance):
    """Use EM to fit a Dirichlet mixture."""

    # hackishly regularize our input vectors
    vectors = vectors + 1e-6
    vectors /= numpy.sum(vectors, axis = -1)[..., None]

    # then do EM
    N = vectors.shape[0]
    components = vectors[numpy.random.randint(N, size = K)]
    responsibilities = numpy.empty((K, N))
    old_ll = -numpy.inf

    for i in xrange(512):
        # compute new responsibilities
        raw_mass = dirichlet_log_pdf(vectors[None, ...], components[:, None, ...])
        log_mass = numpy.sum(raw_mass, axis = 2)
        responsibilities = log_mass - numpy.logaddexp.reduce(log_mass, axis = 0)
        responsibilities = numpy.exp(responsibilities)
        weights = numpy.sum(responsibilities, axis = 1)
        weights /= numpy.sum(weights)
        ll = numpy.sum(numpy.logaddexp.reduce(numpy.log(weights)[:, None] + log_mass, axis = 0))

        logger.info("l-l at iteration %i is %f", i, ll)

        # compute new components
        for k in xrange(K):
            components[k] = fit_dirichlet_vfixed(vectors, responsibilities[k], variance)

        #with numpy_printing(precision = 2, suppress = True, linewidth = 160):
            #print components

        # check for termination
        if numpy.abs(ll - old_ll) <= 1e-3:
            break

        old_ll = ll

    return components

def counts_from_paths(solvers, budgets, paths):
    """Build success/attempt matrices from records."""

    successes = numpy.empty((len(paths), len(solvers), len(budgets)))
    attempts = numpy.empty((len(paths), len(solvers), len(budgets)))

    for (i, path) in enumerate(paths):
        (runs,) = borg.portfolios.get_task_run_data([path]).values()
        (runs_successful, runs_attempted, _) = \
            borg.portfolios.action_rates_from_runs(
                solvers,
                budgets,
                runs.tolist(),
                )

        successes[i] = runs_successful
        attempts[i] = runs_attempted

    return (successes[i], attempts[i])

class BilevelModel(object):
    """Two-level mixture model."""

    def __init__(self, successes, attempts):
        """Fit the model to data."""

        self._successes = successes
        self._attempts = attempts

        # fit the per-solver mixture models
        K = 6
        (N, S, B) = successes.shape
        task_mixes_NSK = numpy.empty((N, S, K))
        self._inner_SKB = numpy.empty((S, K, B))

        for s in xrange(observed.shape[1]):
            (self._inner_SKB[s], _, responsibilities, _) = \
                max(
                    (fit_binomial_mixture(observed[:, s], counts[:, s], K) for _ in xrange(4)),
                    key = lambda (_, __, ___, ll): ll,
                    )
            task_mixes[:, s] = responsibilities.T

            print self._solver_rindex[s]
            print_probability_matrix(self._inner_components[s])

        # fit the outer dirichlet mixture
        L = 8
        self._outer = fit_dirichlet_mixture(task_mixes, L, 100)

        with numpy_printing(precision = 2, suppress = True, linewidth = 160):
            print self._outer_components

