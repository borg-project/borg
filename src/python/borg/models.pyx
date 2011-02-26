"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import scipy.stats
import scikits.learn.linear_model
import numpy
import borg
import cargo

cimport numpy

logger = cargo.get_logger(__name__, default_level = "DETAIL")

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
    """Compute the DCM PDF."""

    sum_alpha = numpy.sum(alpha, axis = -1)
    sum_vector = numpy.sum(vector, axis = -1)

    term_l = scipy.special.gamma(sum_alpha) / scipy.special.gamma(sum_alpha + sum_vector)
    term_r = numpy.prod(scipy.special.gamma(vector + alpha) / scipy.special.gamma(alpha), axis = -1)

    return term_l * term_r

def dcm_draw_pdf(k, alpha):
    """Compute the DCM PDF of a single draw."""

    sum_alpha = numpy.sum(alpha, axis = -1)
    alpha_plus = numpy.copy(alpha)

    alpha_plus[k] += 1.0

    term_l = scipy.special.gamma(sum_alpha) / scipy.special.gamma(sum_alpha + 1.0)
    term_r = numpy.prod(scipy.special.gamma(alpha_plus) / scipy.special.gamma(alpha), axis = -1)

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

    return (successes, attempts)

def assert_probabilities(array):
    assert numpy.all(array >= 0.0)
    assert numpy.all(array <= 1.0)

def assert_weights(array, axis = None):
    assert numpy.all(numpy.abs(numpy.sum(array, axis = axis) - 1.0 ) < 1e-6)

class BilevelModel(object):
    """Two-level mixture model."""

    def __init__(self, successes, attempts):
        """Fit the model to data."""

        # mise en place
        (N, S, B) = successes.shape

        successes_NSB = successes
        attempts_NSB = attempts

        # fit the per-solver mixture models
        K = 6
        task_mixes_NSK = numpy.empty((N, S, K))
        self._inner_SKB = numpy.empty((S, K, B))

        for s in xrange(S):
            fit = lambda: fit_binomial_mixture(successes_NSB[:, s], attempts_NSB[:, s], K)
            (self._inner_SKB[s], _, responsibilities_KN, _) = \
                max(
                    [fit() for _ in xrange(4)],
                    key = lambda x: x[-1],
                    )
            task_mixes_NSK[:, s] = responsibilities_KN.T

            logger.detail(
                "behavior class %i:\n%s",
                s,
                cargo.pretty_probability_matrix(self._inner_SKB[s]),
                )

        # fit the outer dirichlet mixture
        L = 8
        self._outer_LSK = fit_dirichlet_mixture(task_mixes_NSK, L, 100)

        with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 160):
            logger.detail("task classes:\n%s", str(self._outer_LSK))

    def predict(self, failures):
        """Return probabilistic predictions of success."""

        print "failures:", failures

        # mise en place
        F = len(failures)
        (L, S, K) = self._outer_LSK.shape

        # compute task class likelihoods
        tclass_weights_L = numpy.zeros(L)

        for l in xrange(L):
            tclass_weights_L[l] += self._lnp_failures_tclass(failures, l)

        tclass_weights_L = numpy.exp(tclass_weights_L - numpy.logaddexp.reduce(tclass_weights_L))

        with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 160):
            logger.detail("tclass weights:\n%s", str(tclass_weights_L))

        assert_probabilities(tclass_weights_L)
        assert_weights(tclass_weights_L)

        # condition the task classes
        preconditioning_FSK = numpy.ones((F, S, K))

        for f in xrange(F):
            (s, b) = failures[f]

            for k in xrange(K):
                preconditioning_FSK[f, s, k] = 1.0 - self._inner_SKB[s, k, b]

        preconditioning_FSK /= numpy.sum(preconditioning_FSK, axis = -1)[..., None]
        conditioning_SK = numpy.sum(preconditioning_FSK, axis = 0)
        conditioned_LSK = self._outer_LSK + conditioning_SK[None, ...]

        with cargo.numpy_printing(precision = 2, linewidth = 160):
            #logger.detail("preconditioning_FSK:\n%s", str(preconditioning_FSK))
            logger.detail("conditioning_SK:\n%s", str(conditioning_SK))

        # compute posterior probabilities
        tclass_means_LSK = conditioned_LSK / numpy.sum(conditioned_LSK, axis = -1)[..., None]
        tclass_rates_LSB = numpy.sum(tclass_means_LSK[..., None] * self._inner_SKB[None, ...], axis = -2)
        posterior_mean_SK = numpy.sum(tclass_weights_L[:, None, None] * tclass_means_LSK, axis = 0)
        posterior_rates_SB = numpy.sum(posterior_mean_SK[..., None] * self._inner_SKB, axis = 1)

        with cargo.numpy_printing(precision = 2, suppress = True, linewidth = 240):
            logger.detail("tclass_rates_LSB:\n%s", str(tclass_rates_LSB))

        #logger.detail(
            #"posterior probabilities:\n%s",
            #cargo.pretty_probability_matrix(posterior_rates_SB),
            #)

        return (posterior_rates_SB, tclass_weights_L, tclass_rates_LSB)

    def _lnp_failures_tclass(self, failures, l):
        """Return p(failures | task class l)."""

        return sum(self._lnp_failure_tclass(failure, l) for failure in failures)

    def _lnp_failure_tclass(self, failure, l):
        """Return p(s@c failed | task class l)."""

        (s, b) = failure
        (_, _, K) = self._outer_LSK.shape
        sigma = -numpy.inf

        for k in xrange(K):
            lnp_l = numpy.log(1.0 - self._inner_SKB[s, k, b])
            lnp_r = numpy.log(dcm_draw_pdf(k, self._outer_LSK[l, s]))
            sigma = numpy.logaddexp(sigma, lnp_l + lnp_r)

        return sigma

