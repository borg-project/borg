"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.log   import get_logger
from cargo.sugar import composed

log = get_logger(__name__)

def featured_model(feature_count, solver_count, K):
    """
    XXX
    """

    from cargo.statistics import (
        Tuple,
        Binomial,
        MixedBinomial,
        FiniteMixture,
        )

    return \
        FiniteMixture(
            Tuple([
                (Binomial(estimation_n = 1), feature_count),
                (MixedBinomial()           , solver_count ),
                ]),
            K,
            )

class FeaturedMixture(object):
    """
    Feature-aware latent class model of solver and feature outcomes.
    """

    def __init__(
        self                  ,
        feature_actions       ,
        solver_actions        ,
        K                     ,
        concentration   = 1e-8,
        initializations = 1   ,
        ):
        """
        Initialize.
        """

        # build the model
        self.feature_actions = feature_actions
        self.solver_actions  = solver_actions
        self.model           = featured_model(len(feature_actions), len(solver_actions), K)
        self.concentration   = concentration
        self.initializations = initializations

    def feature_indices(self, actions):
        """
        Return the indices corresponding to the specified feature actions.
        """

        return map(self.feature_actions.index, actions)

    def solver_indices(self, actions):
        """
        Return the indices corresponding to the specified solver actions.
        """

        return map(self.solver_actions.index, actions)

    def fetch_samples_array(self, trainer):
        """
        Build an array of action outcomes.
        """

        log.detail(
            "retrieving training data for %i feature and %i solver actions",
            len(self.feature_actions),
            len(self.solver_actions),
            )

        @composed(list)
        def get_samples():
            feature_outcomes = trainer.get_data(self.feature_actions)
            solver_outcomes  = trainer.get_data(self.solver_actions)

            for task in trainer.tasks:
                yield (
                    [feature_outcomes[(a, task.uuid)] for a in self.feature_actions],
                    [solver_outcomes [(a, task.uuid)] for a in self.solver_actions ],
                    )

        return numpy.array(get_samples(), self.model.sample_dtype)

    def train(self, samples):
        """
        Fit the model to data.
        """

        from cargo.statistics import ModelEngine

        log.detail("fitting model parameters to %i training samples", len(samples))

        prior = [(
            [(1.0 + self.concentration,) * 2] * len(self.feature_actions),
            [(1.0 + self.concentration,) * 2] * len(self.solver_actions ),
            )]

        self.parameter = \
            ModelEngine(self.model).map(
                prior,
                samples,
                numpy.ones(len(samples)),
                initializations = self.initializations,
                )

    def submixture(self, some_feature_actions, some_solver_actions):
        """
        Build a partial model.
        """

        mixture = \
            FeaturedMixture(
                some_feature_actions,
                some_solver_actions,
                self.model.K,
                self.concentration,
                self.initializations,
                )

        theta = numpy.empty((), mixture.model.parameter_dtype)

        theta["p"]       = self.parameter["p"]
        theta["c"]["d0"] = self.parameter["c"]["d0"][..., self.feature_indices(mixture.feature_actions)]
        theta["c"]["d1"] = self.parameter["c"]["d1"][..., self.solver_indices(mixture.solver_actions  )]

        mixture.parameter = theta

        return mixture

    def subsamples(self, mixture, samples):
        """
        Build a partial samples array.
        """

        sub = numpy.empty(samples.shape, mixture.model.sample_dtype)

        sub["d0"] = samples["d0"][..., self.feature_indices(mixture.feature_actions)]
        sub["d1"] = samples["d1"][..., self.solver_indices(mixture.solver_actions  )]

        return sub

    def subcondition(self, samples, some_feature_actions, some_solver_actions):
        """
        Return the posterior mixture component weights.
        """

        # compute the posterior component probabilities
        post_pi = numpy.empty((samples.shape[0],) + self.parameter["p"].shape)

        if len(some_feature_actions) + len(some_solver_actions) > 0:
            # condition the model on available data
            from cargo.statistics import ModelEngine

            submixture = self.submixture(some_feature_actions, some_solver_actions)
            subsamples = self.subsamples(submixture, samples)

            post_pi[:] = ModelEngine(submixture.model).given(submixture.parameter, subsamples[..., None])["p"]
        else:
            # no data is available
            post_pi[:] = self.parameter["p"]

        return post_pi

    #def predict(self, post_thetas):
        #"""
        #Return the marginal solver outcome probabilities.
        #"""

        ## XXX not right; we want to marginalize
        #post_outcomes["d0"] = past_samples["d1"]

        #lls = ModelEngine(post_model).ll(post_thetas, post_outcomes)

        #if first_lls is None:
            #first_lls = lls

class FeaturedStrategy(object):
    """
    Feature-aware portfolio strategy.
    """

    def __init__(self, mixture):
        """
        Initialize.
        """

        self._mixture = mixture

    def reset(self):
        """
        Prepare to solve a new task.
        """

        del self._feature_history[:]
        del self._solver_history[:]

    def see(self, action, outcome):
        """
        Witness the outcome of an action.
        """

        if action in self._mixture.feature_actions:
            self._feature_history.append((action, outcome))
        elif action in self._mixture.solver_actions:
            self._solver_history.append((action, outcome))
        else:
            raise ValueError("where did this action come from?")

    def choose(self, budget, random):
        """
        Return the selected action.
        """

        # XXX call the mixture model
        # XXX discounting and argmax

    def _make_samples_array(self):
        """
        Build an array of past action outcomes.
        """

        @composed(list)
        def get_samples():
            feature_outcomes = trainer.get_data(feature_actions)
            solver_outcomes  = trainer.get_data(solver_actions)

            for task in trainer.tasks:
                yield (
                    # XXX
                    [feature_history[(a, task.uuid)] for a in feature_actions],
                    [solver_outcomes [(a, task.uuid)] for a in solver_actions ],
                    )

        return numpy.array(get_samples(), self._model.sample_dtype)

