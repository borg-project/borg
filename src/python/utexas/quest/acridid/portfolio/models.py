"""
utexas/papers/nips2009/models.py

Various models of task/action outcomes.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.log import get_logger
# from cargo.statistics.dcm import DirichletCompoundMultinomial
# from cargo.statistics.mixture import FiniteMixture
from utexas.quest.acridid.portfolio.strategies import ActionModel

log = get_logger(__name__)

class MultinomialActionModel(ActionModel):
    """
    A simple non-mixture multinomial model.
    """

    def __init__(self, world, training):
        """
        Initialize.
        """

        counts       = world.counts_from_events(training)
        total_counts = numpy.sum(counts, 0)
        norm         = numpy.sum(total_counts, 1, numpy.double)[:, numpy.newaxis]

        self.prediction                 = total_counts / norm
        self.prediction.flags.writeable = False

    def predict(self, task, history, out = None):
        """
        Return the predicted probability of each outcome of each action given history.
        """

        if out is None:
            out = self.prediction
        else:
            out[:] = self.prediction

        return out

'''
class MultinomialMixtureActionModel(ActionModel):
    """
    An arbitrary mixture model.
    """

    def __init__(self, world, training, estimator):
        """
        Initialize.
        """

        # members
        self.__world = world
        self.__training = training
        self.__estimator = estimator

        # model
        counts = training.get_positive_counts()
        training_split = [counts[:, naction, :] for naction in xrange(world.nactions)]

        self.__mixture = estimator.estimate(training_split)

    def predict(self, task, history, out = None):
        """
        Return the predicted probability of each outcome given history.
        """

        # mise en place
        M = self.__mixture.ndomains
        K = self.__mixture.ncomponents

        # get the task-specific history
        task_history = history.counts[task.n]

        # evaluate the new responsibilities
        post_pi_K = numpy.copy(self.__mixture.pi)

        for k in xrange(K):
            for m in xrange(M):
                ll = self.__mixture.components[m, k].log_likelihood(task_history[m])

                post_pi_K[k] *= numpy.exp(ll)

        post_pi_K /= numpy.sum(post_pi_K)

        # calculate the expected utility of each action
        if out is None:
            out = numpy.zeros((self.__world.nactions, self.__world.noutcomes))
        else:
            out[:] = 0.0

        v = numpy.empty(self.__world.noutcomes, numpy.uint)

        for o in xrange(self.__world.noutcomes):
            v[:] = 0
            v[o] = 1

            for m in xrange(M):
                for k in xrange(K):
                    ll = self.__mixture.components[m, k].log_likelihood(v)
                    out[m, o] += post_pi_K[k] * numpy.exp(ll)

        # done
        return out

    # properties
    mixture = property(lambda self: self.__mixture)

class DCM_MixtureActionModel(ActionModel):
    """
    An arbitrary mixture model.
    """

    def __init__(self, world, training, estimator):
        """
        Initialize.
        """

        # members
        self.__world = world
        self.__training = training
        self.__estimator = estimator

        # model
        counts = training.get_positive_counts()
        training_split = [counts[:, naction, :] for naction in xrange(world.nactions)]
        self.mixture = estimator.estimate(training_split)

    def predict(self, task, history, out = None):
        """
        Return the predicted probability of each outcome given history.
        """

        # mise en place
        M = self.mixture.ndomains
        K = self.mixture.ncomponents

        # get the task-specific history
        task_history = history.counts[task.n]

        # evaluate the new responsibilities
        post_pi_K = numpy.copy(self.mixture.pi)

        for k in xrange(K):
            for m in xrange(M):
                ll = self.mixture.components[m, k].log_likelihood(task_history[m])

                post_pi_K[k] *= numpy.exp(ll)

        post_pi_K /= numpy.sum(post_pi_K)

        # build the new mixture
        post_components = numpy.empty_like(self.mixture.components)

        for k in xrange(K):
            for m in xrange(M):
                prior = self.mixture.components[m, k]

                post_components[m, k] = DirichletCompoundMultinomial(prior.alpha + task_history[m])

        # calculate the expected utility of each action
        if out is None:
            out = numpy.zeros((self.__world.nactions, self.__world.noutcomes))
        else:
            out[:] = 0.0

        v = numpy.empty(self.__world.noutcomes, numpy.uint)

        for o in xrange(self.__world.noutcomes):
            v[:] = 0
            v[o] = 1

            for m in xrange(M):
                for k in xrange(K):
                    ll = post_components[m, k].log_likelihood(v)
                    out[m, o] += post_pi_K[k] * numpy.exp(ll)

        # done
        return out

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
        self.__last_prediction = (None, None)

    def predict_action(self, task, action):
        """
        Return the predicted probability of C{action} on C{task}.
        """

        counts = numpy.zeros(self.world.noutcomes, dtype = numpy.uint)

        for outcome in self.world.samples.get_outcomes(task, action):
            counts[outcome.n] += 1

        return counts / numpy.sum(counts, dtype = numpy.float)

    def predict(self, task, history, out = None):
        """
        Return the predicted probability of each outcome given history.
        """

        (last_n, last_prediction) = self.__last_prediction

        if last_n == task.n:
            # we already did this
            return last_prediction
        else:
            # calculate the expected utility of each action
            prediction = numpy.empty((self.world.nactions, self.world.noutcomes))

            for action in self.world.actions:
                prediction[action.n] = self.predict_action(task, action)

            # cache
            self.__last_prediction = (task.n, prediction)

            # done
            if out is None:
                return prediction
            else:
                out[:] = prediction

                return out

class RandomActionModel(ActionModel):
    """
    Know nothing.
    """

    def __init__(self, world):
        """
        Initialize.
        """

        # members
        self.world = world

    def predict(self, task, history, out = None):
        """
        Return the predicted probability of each outcome given history.
        """

        if out is None:
            out = numpy.random.random((self.world.nactions, self.world.noutcomes))
        else:
            out[:] = numpy.random.random((self.world.nactions, self.world.noutcomes))

        out /= numpy.sum(out, 1)[:, numpy.newaxis]

        return out

class RankingActionModel(ActionModel):
    """
    Rank actions according to true utility.
    """

    def __init__(self, world, submodel):
        """
        Initialize.
        """

        # members
        self.world = world
        self.submodel = submodel

    def predict(self, task, history, out = None):
        """
        Return the predicted probability of each outcome given history.
        """

        # FIXME use a task_history parameter instead
        # get the task-specific history
        task_history = history.counts[task.n]

#        numpy.uniq?

        if out is None:
            out = FIXME
        else:
            out[:] = FIXME

        return out
'''

