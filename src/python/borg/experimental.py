"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>

Unfinished and/or experimental code. Read at your own risk.
"""

def plan_knapsack_multiverse_speculative(model, failures, budgets, remaining):
    """Multiverse knapsack planner with speculative reordering."""

    # evaluate actions via speculative replanning
    (mean_rates_SB, _, _) = model.predict(failures)
    mean_fail_cmf_SB = numpy.cumprod(1.0 - mean_rates_SB, axis = -1)
    (S, B) = mean_rates_SB.shape
    speculation = []

    for (s, b) in itertools.product(xrange(S), xrange(B)):
        if budgets[b] <= remaining:
            p_f = mean_fail_cmf_SB[s, b]
            (post_plan, post_p_f) = \
                plan_knapsack_multiverse(
                    model,
                    failures + [(s, b)],
                    budgets,
                    remaining - budgets[b],
                    )
            expectation = p_f * post_p_f

            logger.info(
                "action %s p_f = %.2f; post-p_f = %.2f; expectation = %.4f",
                (s, b),
                p_f,
                post_p_f,
                expectation,
                )

            speculation.append(([(s, b)] + post_plan, expectation))

    # move the most useful action first
    if len(speculation) > 0:
        (plan, expectation) = min(speculation, key = lambda (_, e): e)

        return (plan, expectation)
    else:
        return ([], 1.0)

def fit_mixture_model(observed, counts, K):
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
        log_mass = numpy.sum(numpy.sum(raw_mass, axis = 3), axis = 2)
        responsibilities = log_mass - numpy.logaddexp.reduce(log_mass, axis = 0)
        responsibilities = numpy.exp(responsibilities)
        weights = numpy.sum(responsibilities, axis = 1)
        weights /= numpy.sum(weights)
        ll = numpy.sum(numpy.logaddexp.reduce(numpy.log(weights)[:, None] + log_mass, axis = 0))

        logger.debug("l-l at iteration %i is %f", i, ll)

        # compute new components
        map_observed = observed[None, ...] * responsibilities[..., None, None]
        map_counts = counts[None, ...] * responsibilities[..., None, None]
        components = numpy.sum(map_observed, axis = 1) + concentration / 2.0
        components /= numpy.sum(map_counts, axis = 1) + concentration

        # check for termination
        if numpy.abs(ll - old_ll) <= 1e-3:
            break

        old_ll = ll

    assert numpy.all(components >= 0.0)
    assert numpy.all(components <= 1.0)

    return (components, weights, responsibilities, ll)

def format_probability(p):
    return "C " if p >= 0.995 else "{0:02.0f}".format(p * 100.0)

def print_probability_row(row):
    print " ".join(map(format_probability, row))

def print_probability_matrix(matrix):
    for row in matrix:
        print_probability_row(row)

class MixturePortfolio(object):
    """Mixture-model portfolio."""

    def __init__(self, solvers, train_paths):
        # build action set
        self._solvers = solvers
        self._budgets = budgets = [(b + 1) * 100.0 for b in xrange(50)]

        # acquire running time data
        self._solver_rindex = dict(enumerate(solvers))
        self._solver_index = dict(map(reversed, self._solver_rindex.items()))
        budget_rindex = dict(enumerate(budgets))
        self._budget_index = dict(map(reversed, budget_rindex.items()))
        observed = numpy.empty((len(train_paths), len(solvers), len(budgets)))
        counts = numpy.empty((len(train_paths), len(solvers), len(budgets)))

        logger.info("solver index: %s", self._solver_rindex)

        for (i, train_path) in enumerate(train_paths):
            (runs,) = get_task_run_data([train_path]).values()
            (runs_observed, runs_counts, _) = \
                action_rates_from_runs(
                    self._solver_index,
                    self._budget_index,
                    runs.tolist(),
                    )

            observed[i] = runs_observed
            counts[i] = runs_counts

            logger.info("true probabilities for %s:", os.path.basename(train_path))
            print_probability_matrix(observed[i] / counts[i])

        # fit the mixture model
        best = -numpy.inf

        for K in [64] * 4:
            (components, weights, responsibilities, ll) = fit_mixture_model(observed, counts, K)

            logger.info("model with K = %i has ll = %f", K, ll)

            if ll > best:
                self._components = components
                self._weights = weights
                best = ll

        logger.info("fit mixture with best ll = %f", best)

        for i in xrange(K):
            logger.info("mixture component %i (weight %f):", i, self._weights[i])
            print_probability_matrix(self._components[i])

        raise SystemExit()

        # fit the cluster classifiers
        train_x = []
        train_ys = map(lambda _: [], xrange(K))
        raw_mass = scipy.stats.binom.logpmf(observed[None, ...], counts[None, ...], components[:, None, ...])
        mass = numpy.exp(numpy.sum(numpy.sum(raw_mass, axis = 3), axis = 2))

        for (i, train_path) in enumerate(train_paths):
            features = numpy.recfromcsv(train_path + ".features.csv").tolist()

            train_x.extend([features] * 100)

            for k in xrange(K):
                positives = int(round(mass[k, i] * 100))
                negatives = 100 - positives

                train_ys[k].extend([1] * positives)
                train_ys[k].extend([0] * negatives)

        self._classifiers = []

        for k in xrange(K):
            classifier = scikits.learn.linear_model.LogisticRegression()

            classifier.fit(train_x, train_ys[k])

            self._classifiers.append(classifier)

        logger.info("trained the cluster classifiers")

    def __call__(self, cnf_path, budget):
        # obtain features
        csv_path = cnf_path + ".features.csv"

        if os.path.exists(csv_path):
            features = numpy.recfromcsv(csv_path).tolist()
        else:
            (_, features) = borg.features.get_features_for(cnf_path)

        features_cost = features[0]

        # use classifier to establish initial cluster probabilities
        #prior_weights = numpy.empty(self._components.shape[0])

        #for k in xrange(prior_weights.size):
            #prior_weights[k] = self._classifiers[k].predict_proba([features])[0, -1]

        #prior_weights /= numpy.sum(prior_weights)

        prior_weights = self._weights

        print " ".join(map("{0:02.0f}".format, prior_weights * 100))

        # use oracle knowledge to establish initial cluster probabilities
        (runs,) = get_task_run_data([cnf_path]).values()
        (oracle_history, oracle_counts, _) = \
            action_rates_from_runs(
                self._solver_index,
                self._budget_index,
                runs.tolist(),
                )
        true_rates = oracle_history / oracle_counts
        #raw_mass = scipy.stats.binom.logpmf(oracle_history, oracle_counts, self._components)
        #log_mass = numpy.sum(numpy.sum(raw_mass, axis = 2), axis = 1)
        #prior_weights = numpy.exp(log_mass - numpy.logaddexp.reduce(log_mass))

        # select a solver
        initial_utime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        total_run_cost = features_cost
        history = numpy.zeros((len(self._solvers), len(self._budgets)))
        new_weights = prior_weights
        first_rates = numpy.sum(self._components * prior_weights[:, None, None], axis = 0)
        answer = None

        logger.info("solving %s", cnf_path)

        while True:
            # compute marginal probabilities of success
            #overhead = resource.getrusage(resource.RUSAGE_SELF).ru_utime - initial_utime
            overhead = 0.0 # XXX
            total_cost = total_run_cost + overhead

            with fpe_handling(under = "ignore"):
                rates = numpy.sum(self._components * new_weights[:, None, None], axis = 0)

            # generate a plan
            plan = knapsack_plan(rates, self._solver_rindex, self._budgets, budget - total_cost)

            if not plan:
                break

            # XXX if our belief is zero, bail or otherwise switch to plan B

            (name, planned_cost) = plan[0]

            if budget - total_cost - planned_cost < self._budgets[0]:
                max_cost = budget - total_cost
            else:
                max_cost = planned_cost

            # run it
            logger.info(
                "taking %s@%i with %i remaining (b = %.2f; p = %.2f)",
                name,
                max_cost,
                budget - total_cost,
                rates[self._solver_index[name], self._budget_index[planned_cost]],
                true_rates[self._solver_index[name], self._budget_index[planned_cost]],
                )

            # XXX adjust cost
            solver = self._solvers[name]
            (run_cost, answer) = solver(cnf_path, max_cost)
            total_run_cost += run_cost

            if answer is not None:
                break

            # recalculate cluster probabilities
            history[self._solver_index[name]] += 1.0

            raw_mass = scipy.stats.binom.logpmf(numpy.zeros_like(history), history, self._components)
            log_mass = numpy.sum(numpy.sum(raw_mass, axis = 2), axis = 1)
            new_weights = numpy.exp(log_mass - numpy.logaddexp.reduce(log_mass))
            new_weights *= prior_weights
            new_weights /= numpy.sum(new_weights)

            #raw_mass = scipy.stats.binom.logpmf(oracle_history, oracle_counts + history, self._components)
            #log_mass = numpy.sum(numpy.sum(raw_mass, axis = 2), axis = 1)
            #new_weights = numpy.exp(log_mass - numpy.logaddexp.reduce(log_mass))

        total_cost = total_run_cost + overhead

        logger.info("answer %s with total cost %.2f", answer, total_cost)

        if answer is None:
            print self._solver_rindex
            print "truth:"
            print "\n".join(" ".join(map("{0:02.0f}".format, row * 100)) for row in true_rates)
            print "initial beliefs:"
            print "\n".join(" ".join(map("{0:02.0f}".format, row * 100)) for row in first_rates)
            print "final beliefs:"
            print "\n".join(" ".join(map("{0:02.0f}".format, row * 100)) for row in rates)

        return (total_cost, answer)

