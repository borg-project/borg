"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc         import abstractmethod
from cargo.log   import get_logger
from cargo.sugar import ABC
# from cargo.statistics.dcm import DirichletCompoundMultinomial
# from cargo.statistics.mixture import FiniteMixture
# from utexas.portfolio.world import get_positive_counts
from cargo.statistics._statistics import (
    dcm_post_pi_K,
    dcm_model_predict,
    multinomial_model_predict,
    )

log = get_logger(__name__)

def assert_sane_predictions(predictions):
    """
    Assert the a predictions map makes superficial sense.
    """

    for (action, prediction) in predictions.iteritems():
        if len(action.outcomes) != len(prediction):
            raise ValueError("prediction count does not match outcome count")
        if numpy.sum(prediction) != 1.0:
            raise ValueError("non-normalized probability vector")

class ActionModel(ABC):
    """
    A model of action outcomes.
    """

    @abstractmethod
    def predict(self, task, history):
        """
        Return a map between actions and (normalized) predicted probabilities.
        """

class FixedActionModel(ActionModel):
    """
    Stick to one prediction.
    """

    def __init__(self, predictions):
        """
        Initialize.
        """

        # argument sanity
        assert_sane_predictions(predictions)

        # members
        self._predictions = predictions

    def predict(self, task, history):
        """
        Return the fixed map.
        """

        return self._predictions

class RandomActionModel(ActionModel):
    """
    Make random predictions.
    """

    def __init__(self, actions, random = numpy.random):
        """
        Initialize.
        """

        self.actions = actions
        self.random  = random

    def _random_prediction(self, action):
        """
        Return a random prediction.
        """

        predictions  = self.random.rand(len(action.outcomes))
        predictions /= numpy.sum(predictions)

        return predictions

    def predict(self, task, history):
        """
        Return the predicted probability of each outcome, given history.
        """

        return dict((a, self._random_prediction(a)) for a in self.actions)

class MultinomialMixtureActionModel(ActionModel):
    """
    An arbitrary mixture model.
    """

    # FIXME why do we *accept* an estimator as a parameter? build it here.
    # FIXME (or make a general distribution action model...)

    def __init__(self, training, estimator):
        """
        Initialize.

        @param training: [tasks-by-outcomes counts array for each action]
        """

        # model
        self._mixture = estimator.estimate(training.values())
        self._actions = training.keys()

        # store mixture components in a matrix
        M = self.mixture.ndomains
        K = self.mixture.ncomponents

#         self.mix_KD_per = numpy.empty((M, K), numpy.object)

#         for m in xrange(M):
#             for k in xrange(K):
#                 self.mix_KD_per[m][k] = self.mixture.components[m, k].log_beta
        D = 2 # FIXME

        self.mix_MKD = numpy.empty((M, K, D))

        for m in xrange(M):
            for k in xrange(K):
                self.mix_MKD[m, k] = self.mixture.components[m, k].log_beta

    def predict(self, task, history, feasible):
        """
        Given history, return the probability of each outcome of each action.

        @param history: [outcome counts array for each action]
        @return: [outcome probabilities array for each action]
        """

        # mise en place
        M = self.mixture.ndomains
        K = self.mixture.ncomponents
        D = 2 # FIXME

        action_indices = dict((a, i) for (i, a) in enumerate(self._actions))

        # get the task-specific history
        counts_MD = numpy.zeros((len(self._actions), 2), numpy.uint)

        for (t, a, o) in history:
            if t == task:
                na = action_indices[a]
                counts_MD[na, o.n] += 1

        # get the outcome probabilities
        out = numpy.empty((M, D))
        pi  = numpy.copy(self._mixture.pi)

        multinomial_model_predict(
            pi,
            self.mix_MKD,
            counts_MD,
            out,
            )

        # log an outcome-probability table
        rows = {}

        for action in self._actions:
            ps = rows.get(action.solver, [])

            ps.append((action.cutoff, out[action_indices[action]]))

            rows[action.solver] = ps

        sorted_rows = [(k.name, sorted(v, key = lambda (c, p): c)) for (k, v) in rows.items()]
        sorted_all  = sorted(sorted_rows, key = lambda (k, v): k)
        longest     = max(len(s) for (s, _) in sorted_all)
        table       = \
            "\n".join(
                "%s: %s" % (s.ljust(longest + 1), " ".join("%.4f" % p[0] for (c, p) in r)) \
                for (s, r) in sorted_all \
                )

        log.debug("probabilities of action success (multinomial model):\n%s", table)

        # return predictions
        predicted = out[numpy.array([action_indices[a] for a in feasible])]

        return dict(zip(feasible, predicted))

    # properties
    mixture = property(lambda self: self._mixture)

class DCM_MixtureActionModel(ActionModel):
    """
    A DCM mixture model.
    """

    def __init__(self, training, estimator):
        """
        Initialize.
        """

        # model
        self.mixture  = estimator.estimate(training.values())
        self._actions = training.keys()

        # cache mixture components as matrices
        M = self.mixture.ndomains
        K = self.mixture.ncomponents
        D = 2 # FIXME

        self.sum_MK  = numpy.empty((M, K))
        self.mix_MKD = numpy.empty((M, K, D))

        for m in xrange(M):
            for k in xrange(K):
                component          = self.mixture.components[m, k]
                self.sum_MK[m, k]  = component.sum_alpha
                self.mix_MKD[m, k] = component.alpha

    def get_post_pi_K(self, counts_MD):

        post_pi_K = numpy.copy(self.mixture.pi)

        dcm_post_pi_K(
            post_pi_K,
            self.sum_MK,
            self.mix_MKD,
            counts_MD,
            )

        return post_pi_K

    def predict(self, task, history, feasible):

        # mise en place
        M = self.mixture.ndomains
        K = self.mixture.ncomponents
        D = 2 # FIXME

        action_indices = dict((a, i) for (i, a) in enumerate(self._actions))

        # get the task-specific history
        counts_MD = numpy.zeros((len(self._actions), 2), numpy.uint)

        for (t, a, o) in history:
            if t == task:
                na = action_indices[a]
                counts_MD[na, o.n] += 1

        # get the outcome probabilities
        post_pi_K = self.get_post_pi_K(counts_MD)
        out       = numpy.empty((M, D))

        dcm_model_predict(
            post_pi_K,
            numpy.copy(self.sum_MK),
            numpy.copy(self.mix_MKD),
            counts_MD,
            out,
            )

        # log an outcome-probability table
        rows = {}

        for action in self._actions:
            ps = rows.get(action.solver, [])

            ps.append((action.cutoff, out[action_indices[action]]))

            rows[action.solver] = ps

        sorted_rows = [(k.name, sorted(v, key = lambda (c, p): c)) for (k, v) in rows.items()]
        sorted_all  = sorted(sorted_rows, key = lambda (k, v): k)
        longest     = max(len(s) for (s, _) in sorted_all)
        table       = \
            "\n".join(
                "%s: %s" % (s.ljust(longest + 1), " ".join("%.4f" % p[0] for (c, p) in r)) \
                for (s, r) in sorted_all \
                )

        log.debug("probabilities of action success (DCM model):\n%s", table)

        # return predictions
        predicted = out[numpy.array([action_indices[a] for a in feasible])]

        return dict(zip(feasible, predicted))

class OracleActionModel(ActionModel):
    """
    Nosce te ipsum.
    """

    def __init__(self, world):
        """
        Initialize.
        """

        # members
        self.world = world

        # FIXME hack
        self.last_task_n = None

    def predict(self, task, history, out = None):
        """
        Return the predicted probability of each outcome given history.
        """

        # FIXME obviously a hack
        if self.last_task_n == task.n:
            ps = self.last_ps
        else:
            # FIXME obviously inefficient
            ps               = numpy.array([self.world.get_true_probabilities(task, a) for a in self.world.actions])
            self.last_ps     = ps
            self.last_task_n = task.n

        if out is None:
            return ps
        else:
            out[:] = ps

        return out

