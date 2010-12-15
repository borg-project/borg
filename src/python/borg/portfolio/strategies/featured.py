"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.log   import get_logger
from cargo.sugar import composed

log = get_logger(__name__)

def featured_model(feature_count, solver_count, K):
    """
    Build a mixture model as specified.
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

        d0_indices = self.feature_indices(mixture.feature_actions)
        d1_indices = self.solver_indices(mixture.solver_actions  )

        theta["p"] = self.parameter["p"]

        if len(d0_indices) > 0:
            theta["c"]["d0"] = self.parameter["c"]["d0"][..., d0_indices]

        if len(d1_indices) > 0:
            theta["c"]["d1"] = self.parameter["c"]["d1"][..., d1_indices]

        mixture.parameter = theta

        return mixture

    def subsamples(self, mixture, samples):
        """
        Build a partial samples array.
        """

        sub = numpy.empty(samples.shape, mixture.model.sample_dtype)

        d0_indices = self.feature_indices(mixture.feature_actions)
        d1_indices = self.solver_indices(mixture.solver_actions  )

        if len(d0_indices) > 0:
            sub["d0"] = samples["d0"][..., d0_indices]

        if len(d1_indices) > 0:
            sub["d1"] = samples["d1"][..., d1_indices]

        return sub

    def subcondition(self, samples, some_feature_actions, some_solver_actions):
        """
        Return the posterior mixture component weights.
        """

        # compute the posterior component probabilities
        post = numpy.empty(len(samples) + self.parameter["p"].shape)

        if len(some_feature_actions) + len(some_solver_actions) > 0:
            # condition the model on available data
            from cargo.statistics import ModelEngine

            submixture = self.submixture(some_feature_actions, some_solver_actions)
            subsamples = self.subsamples(submixture, samples)

            given = ModelEngine(submixture.model).given

            post[:] = given(submixture.parameter, subsamples[..., None])["p"]
        else:
            # no data is available
            post[:] = self.parameter["p"]

        return post

    def solver_marginals(self):
        """
        Return the marginal solver outcome probabilities.
        """

        return numpy.sum(self.parameter["p"][..., None] * self.parameter["c"]["d1"], 0)

class FeaturedStrategy(object):
    """
    Feature-aware portfolio strategy.
    """

    def __init__(self, mixture):
        """
        Initialize.
        """

        self.mixture = mixture
        self._feature_history = {}
        self._solver_history = {}

    def reset(self):
        """
        Prepare to solve a new task.
        """

        self._feature_history.clear()
        self._solver_history.clear()

    def see(self, action, outcome):
        """
        Witness the outcome of an action.
        """

        if action in self.mixture.feature_actions:
            if action in self._feature_history:
                raise RuntimeError("repeated feature action")
            else:
                self._feature_history[action] = action.outcomes.index(outcome)
        elif action in self.mixture.solver_actions:
            if action in self._solver_history:
                (k, n) = self._solver_history[action]
            else:
                k = 0
                n = 0

            j = 1 if outcome.utility > 0.0 else 0

            self._solver_history[action] = (k + j, n + 1)
        else:
            raise ValueError("action of unknown origin")

    def choose(self, budget, random):
        """
        Return the selected action.
        """

        # first collect features
        for action in self.mixture.feature_actions:
            if action not in self._feature_history:
                return action

        # then run solvers
        if len(self._feature_history) + len(self._solver_history) > 0:
            # condition the model on history
            from cargo.statistics import ModelEngine

            post       = self.mixture.submixture([], self.mixture.solver_actions)
            submixture = \
                self.mixture.submixture(
                    self._feature_history.keys(),
                    self._solver_history.keys(),
                    )
            samples = [(
                self._feature_history.values(),
                self._solver_history.values(),
                )]

            post.parameter["p"] = submixture.model.posterior(submixture.parameter, samples)
        else:
            # no history yet
            post = self.mixture

        # select the best-looking feasible solver
        discounts = numpy.array([0.9999**a.cost for a in self.mixture.solver_actions])
        marginals = post.solver_marginals() * discounts

        for i in reversed(numpy.argsort(marginals)):
            action = self.mixture.solver_actions[i]

            if action.cost <= budget:
                return action

