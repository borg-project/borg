"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import cargo.statistics

def outcome_model(solver_count, K):
    """Build a mixture model as specified."""

    from cargo.statistics import (
        Tuple,
        MixedBinomial,
        FiniteMixture,
        )

    return \
        FiniteMixture(
            Tuple([
                (MixedBinomial(), solver_count),
                ]),
            K,
            )

class OutcomeMixture(object):
    """Latent class model of solver outcomes."""

    def __init__(
        self                  ,
        solver_actions        ,
        K                     ,
        concentration   = 1e-8,
        initializations = 1   ,
        ):
        self.solver_actions  = solver_actions
        self.model           = featured_model(len(feature_actions), len(solver_actions), K)
        self.concentration   = concentration
        self.initializations = initializations

    def solver_indices(self, actions):
        """
        Return the indices corresponding to the specified solver actions.
        """

        return map(self.solver_actions.index, actions)

    def train(self, samples):
        """
        Fit the model to data.
        """

        log.detail("fitting model parameters to %i training samples", len(samples))

        prior = [([(1.0 + self.concentration,) * 2] * len(self.solver_actions),)]

        engine = cargo.statistics.ModelEngine(self.model)

        self.parameter = \
            engine.map(
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

