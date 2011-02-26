"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import resource
import itertools
import contextlib
import numpy
import scipy.stats
import scikits.learn.linear_model
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def format_probability(p):
    return "C " if p >= 0.995 else "{0:02.0f}".format(p * 100.0)

def print_probability_row(row):
    print " ".join(map(format_probability, row))

def print_probability_matrix(matrix):
    for row in matrix:
        print_probability_row(row)

@contextlib.contextmanager
def numpy_printing(**kwargs):
    old = numpy.get_printoptions()

    numpy.set_printoptions(**kwargs)

    yield

    numpy.set_printoptions(**old)

class BilevelPortfolio(object):
    """Bilevel mixture-model portfolio."""

    def __init__(self, solvers, train_paths):
        # build action set
        self._solvers = solvers
        self._budgets = budgets = [(b + 1) * 200.0 for b in xrange(25)]

        # acquire running time data
        self._solver_rindex = dict(enumerate(solvers))
        self._solver_index = dict(map(reversed, self._solver_rindex.items()))
        self._budget_rindex = dict(enumerate(budgets))
        self._budget_index = dict(map(reversed, self._budget_rindex.items()))

        (successes, attempts) = borg.models.counts_from_paths(train_paths)

    def __call__(self, cnf_path, budget):
        # gather oracle knowledge
        (runs,) = borg.portfolios.get_task_run_data([cnf_path]).values()
        (oracle_history, oracle_counts, _) = \
            borg.portfolios.action_rates_from_runs(
                self._solver_index,
                self._budget_index,
                runs.tolist(),
                )
        true_rates = oracle_history / oracle_counts

        print "true probabilities"
        print_probability_matrix(true_rates)

        # select a solver
        total_run_cost = 0.0
        failures = numpy.zeros((len(self._solvers), len(self._budgets)))
        answer = None

        logger.info("solving %s", os.path.basename(cnf_path))

        while True:
            # compute marginal probabilities of success
            # XXX ignoring prior weights
            raw_mass = scipy.stats.binom.logpmf(numpy.zeros_like(failures)[:, None, :], failures[:, None, :], self._inner_components)
            log_mass = numpy.sum(raw_mass, axis = 2)
            run_weights = numpy.exp(log_mass - numpy.logaddexp.reduce(log_mass, axis = 1)[:, None])
            conditioning = run_weights * numpy.sum(failures, axis = 1)[:, None]

            with numpy_printing(precision = 2, suppress = True):
                print "run_weights:"
                print run_weights

            #for l in xrange(L):
                #for k in xrange(K):

                #type_weights[l] = XXX

            type_pdfs = dcm_pdf(conditioning, self._outer_components)

            type_weights = numpy.prod(type_pdfs, axis = -1)
            type_weights /= numpy.sum(type_weights)

            with numpy_printing(precision = 2, suppress = True):
                print "conditioning:"
                print conditioning
                print "type_pdfs:"
                print type_pdfs
                print "type_weights:"
                print type_weights

            conditioned = self._outer_components + conditioning[None, ...]
            normalized = conditioned / numpy.sum(conditioned, axis = -2)[:, None]
            outer_weighted = numpy.sum(normalized * type_weights[:, None, None], axis = 0)
            outer_weighted /= numpy.sum(outer_weighted, axis = -1)[:, None]
            inner_weighted = outer_weighted[..., None] * self._inner_components
            rates = numpy.sum(inner_weighted, axis = 1)

            with numpy_printing(precision = 2, suppress = True):
                print "outer_weighted:"
                print outer_weighted

            #print "reweighted:"
            #for s in xrange(reweighted.shape[0]):
                #print self._solver_rindex[s]
                #print_probability_matrix(reweighted[s])

            print "posterior probabilities:"
            print_probability_matrix(rates)

            # generate a plan
            total_cost = total_run_cost

            plan = borg.portfolios.knapsack_plan(rates, self._solver_rindex, self._budgets, budget - total_cost)

            if not plan:
                break

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

            solver = self._solvers[name]
            (run_cost, answer) = solver(cnf_path, max_cost)
            total_run_cost += run_cost

            if answer is not None:
                break
            else:
                failures[self._solver_index[name], self._budget_index[planned_cost]] += 1.0

        logger.info("answer %s with total cost %.2f", answer, total_cost)

        if answer is None:
            raise SystemExit()
            #print self._solver_rindex
            #print "truth:"
            #print "\n".join(" ".join(map("{0:02.0f}".format, row * 100)) for row in true_rates)
            #print "initial beliefs:"
            #print "\n".join(" ".join(map("{0:02.0f}".format, row * 100)) for row in first_rates)
            #print "final beliefs:"
            #print "\n".join(" ".join(map("{0:02.0f}".format, row * 100)) for row in rates)

        return (total_cost, answer)

borg.portfolios.named["bilevel"] = BilevelPortfolio

