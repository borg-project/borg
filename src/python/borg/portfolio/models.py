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

log = get_logger(__name__)

def assert_sane_predictions(predictions):
    """
    Assert that a predictions map makes superficial sense.
    """

    for row in predictions:
        if numpy.sum(row) != 1.0:
            raise ValueError("non-normalized probability vector")

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

        predictions  = random.rand(*history.shape)
        predictions /= numpy.sum(predictions, 1)[:, None]

        return predictions

    @property
    def actions(self):
        """
        The actions associated with this model.
        """

        return self._actions

class DistributionModel(AbstractModel):
    """
    A general conditional-prediction model.

    Compatible with any distribution over sequences of per-action outcome
    counts, eg, NM-shaped uint arrays.
    """

    def __init__(self, distribution, actions):
        """
        Initialize.
        """

        self._distribution = distribution
        self._actions      = actions

    def predict(self, history, random):
        """
        Return a prediction.
        """

        posterior = self._distribution.given([history])
        indicator = numpy.zeros(history.shape, numpy.uint)
        predicted = numpy.empty(history.shape)

        for i in xrange(predicted.shape[0]):
            for j in xrange(predicted.shape[1]):
                indicator[i, j] = 1
                predicted[i, j] = numpy.exp(posterior.log_likelihood(indicator))
                indicator[i, j] = 0

        predicted /= numpy.sum(predicted, 1)[:, None]

        return predicted

    @property
    def actions(self):
        """
        The actions associated with this model.
        """

        return self._actions

