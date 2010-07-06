"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc         import (
    abstractmethod,
    abstractproperty,
    )
from cargo.log   import get_logger
from cargo.sugar import ABC
from cargo.statistics._statistics import (
    multinomial_model_predict,
    )

log = get_logger(__name__)

def assert_sane_predictions(predictions):
    """
    Assert that a predictions map makes superficial sense.
    """

    for row in predictions:
        if numpy.sum(row) != 1.0:
            raise ValueError("non-normalized probability vector")

def build_model(request, trainer):
    """
    Build a model as requested.
    """

    builders = {
        "oracle"      : OracleModel.build,
        "dcm"         : DCM_MixtureModel.build,
        "multinomial" : MultinomialMixtureModel.build,
        "random"      : RandomModel.build,
        "fixed"       : FixedModel.build,
        "pickle"      : load_model,
        }

    return builders[request["type"]](request, trainer)

def load_model(request, trainer):
    """
    Load a model as requested.
    """

    import cPickle as pickle

    from cargo.io import expandpath

    with open(expandpath(request["path"])) as file:
        return pickle.load(file)

class AbstractModel(ABC):
    """
    A model of action outcomes.
    """

    @abstractmethod
    def predict(self, history, random):
        """
        Return an array of per-action (normalized) outcome probabilities.
        """

    @abstractproperty
    def actions(self):
        """
        The actions associated with this model.
        """

    @property
    def predictor(self):
        """
        Get the fast predictor associated with this model, if any.
        """

        return None

class FixedModel(AbstractModel):
    """
    Stick to one prediction.
    """

    def __init__(self, actions, predictions):
        """
        Initialize.
        """

        # argument sanity
        assert_sane_predictions(predictions)

        if len(actions) != predictions.shape[0]:
            raise ValueError("actions/predictions shape mismatch")

        # members
        self._actions     = actions
        self._predictions = predictions

    def predict(self, history, random):
        """
        Return the fixed map.
        """

        return self._predictions

    @property
    def actions(self):
        """
        The actions associated with this model.
        """

        return self._actions

    @staticmethod
    def build(request, trainer):
        """
        Build a model as requested.
        """

        raise NotImplementedError()

class RandomModel(AbstractModel):
    """
    Make random predictions.
    """

    def __init__(self, actions):
        """
        Initialize.
        """

        self._actions = actions

    def predict(self, history, random):
        """
        Return the predicted probability of each outcome, given history.
        """

        predictions = \
            numpy.empty(
                (
                    len(self._actions),
                    max(len(a.outcomes) for a in self._actions),
                    ),
                )

        for (i, a) in enumerate(self._actions):
            p  = random.rand(len(a.outcomes))
            p /= numpy.sum(p)

            predictions[i, :p.size] = p

        return predictions

    @property
    def actions(self):
        """
        The actions associated with this model.
        """

        return self._actions

    @staticmethod
    def build(request, trainer):
        """
        Build a model as requested.
        """

        return RandomModel(trainer.build_actions(request["actions"]))

class MultinomialMixtureModel(AbstractModel):
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

    def predict(self, history, feasible):
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

            ps.append((action.cost, out[action_indices[action]]))

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

    @staticmethod
    def build_with(training, k, em_restarts):
        """
        Build a model as specified.
        """

        log.info("building a multinomial mixture model")

        from cargo.statistics.mixture     import (
            RestartedEstimator,
            EM_MixtureEstimator,
            smooth_multinomial_mixture,
            )
        from cargo.statistics.multinomial import MultinomialEstimator
        from borg.portfolio.models        import MultinomialMixtureActionModel

        model = \
            MultinomialMixtureModel(
                training,
                RestartedEstimator(
                    EM_MixtureEstimator(
                        [[MultinomialEstimator()] * ncomponents] * len(training),
                        ),
                    nrestarts = nrestarts,
                    ),
                )

        smooth_multinomial_mixture(model.mixture)

        return model

    @staticmethod
    def build(request, trainer):
        """
        Build a model as requested.
        """

        # get actions and training samples
        actions = trainer.build_actions(request["actions"])
        samples = dict((a, trainer.get_data(a)) for a in actions)

        # build the model
        return \
            MultinomialMixtureModel.build_with(
                samples,
                request["components"],
                request["em_restarts"],
                )

    # properties
    mixture = property(lambda self: self._mixture)

class DCM_MixtureModel(AbstractModel):
    """
    A DCM mixture model.
    """

    def __init__(self, training, estimator):
        """
        Initialize.
        """

        self.__setstate__((estimator.estimate(training.values()), training.keys()))

    def __getstate__(self):
        """
        Return picklable state.
        """

        return (self.mixture, self._actions)

    def __setstate__(self, (mixture, actions)):
        """
        Apply unpickled state.
        """

        from borg.portfolio._models import DCM_MixturePredictor

        self.mixture    = mixture
        self._actions   = actions
        self._predictor = DCM_MixturePredictor(actions, mixture)

    def predict(self, history, random):
        """
        Return a prediction.
        """

        out = numpy.empty(history.shape)

        self._predictor.predict(
            numpy.ascontiguousarray(history),
            numpy.ascontiguousarray(out),
            )

        return out

    @property
    def actions(self):
        """
        The actions associated with this model.
        """

        return self._actions

    @property
    def predictor(self):
        """
        Get the fast predictor associated with this model, if any.
        """

        return self._predictor

    @staticmethod
    def build_with(training, k, em_restarts):
        """
        Build a model as specified.
        """

        log.info("building a DCM mixture model")

        from cargo.statistics.dcm     import (
            DCM_Estimator,
            smooth_dcm_mixture,
            )
        from cargo.statistics.mixture import (
            RestartedEstimator,
            EM_MixtureEstimator,
            )

        model = \
            DCM_MixtureModel(
                training,
                RestartedEstimator(
                    EM_MixtureEstimator(
                        [[DCM_Estimator()] * k] * len(training),
                        ),
                    nrestarts = em_restarts,
                    ),
                )

        smooth_dcm_mixture(model.mixture)

        return model

    @staticmethod
    def build(request, trainer):
        """
        Build a model as requested.
        """

        # get actions and training samples
        actions = trainer.build_actions(request["actions"])
        samples = dict((a, trainer.get_data(a)) for a in actions)

        # verify samples sanity
        length = None

        for (_, v) in samples.iteritems():
            if length is None:
                length = len(v)
            else:
                assert len(v) == length

        # build the action model
        return \
            DCM_MixtureModel.build_with(
                samples,
                request["components"],
                request["em_restarts"],
                )

class DistributionModel(AbstractModel):
    """
    A general conditional-prediction model.
    """

    def __init__(self, distribution, actions):
        """
        Initialize.
        """

        self.__setstate__((distribution, actions))

    def __getstate__(self):
        """
        Return picklable state.
        """

        return (self._distribution, self._actions)

    def __setstate__(self, (distribution, actions)):
        """
        Apply unpickled state.
        """

        from borg.portfolio._models import DistributionPredictor

        self._distribution = distribution
        self._actions      = actions
        self._predictor    = DistributionPredictor(actions, mixture)

    def predict(self, history, random):
        """
        Return a prediction.
        """

        # essentially:
        # self._distribution.log_likelihood_given([success for every action], history)

        raise NotImplementedError()

    @property
    def actions(self):
        """
        The actions associated with this model.
        """

        return self._actions

    @property
    def predictor(self):
        """
        Get the fast predictor associated with this model, if any.
        """

        return self._predictor

    @staticmethod
    def build_with(training, k, em_restarts):
        """
        Build a model as specified.
        """

        log.info("building a DCM mixture model")

        from cargo.statistics.dcm     import (
            DCM_Estimator,
            smooth_dcm_mixture,
            )
        from cargo.statistics.mixture import (
            RestartedEstimator,
            EM_MixtureEstimator,
            )

        model = \
            DCM_MixtureModel(
                training,
                RestartedEstimator(
                    EM_MixtureEstimator(
                        [[DCM_Estimator()] * k] * len(training),
                        ),
                    nrestarts = em_restarts,
                    ),
                )

        smooth_dcm_mixture(model.mixture)

        return model

    @staticmethod
    def build(request, trainer):
        """
        Build a model as requested.
        """

        # get actions and training samples
        actions = trainer.build_actions(request["actions"])
        samples = dict((a, trainer.get_data(a)) for a in actions)

        # verify samples sanity
        length = None

        for (_, v) in samples.iteritems():
            if length is None:
                length = len(v)
            else:
                assert len(v) == length

        # build the action model
        return \
            DCM_MixtureModel.build_with(
                samples,
                request["components"],
                request["em_restarts"],
                )

class OracleModel(AbstractModel):
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

    def predict(self, task, history, random):
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

    @staticmethod
    def build(request, trainer):
        """
        Build a model as requested.
        """

        raise NotImplementedError()

