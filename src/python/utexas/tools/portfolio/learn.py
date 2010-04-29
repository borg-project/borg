# vim: set fileencoding=UTF-8 :
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.learn import main

    raise SystemExit(main())

import numpy

from cargo.log import get_logger

log = get_logger(__name__, level = "NOTE")

# portfolio configuration information:
# - model name
# - model configuration
# - planner name
# - planner configuration

def smooth_mult(mixture):
    """
    Apply a smoothing term to the multinomial mixture components.
    """

    from cargo.statistics.multinomial import Multinomial

    log.info("heuristically smoothing mult mixture")

    epsilon = 1e-3

    for m in xrange(mixture.ndomains):
        for k in xrange(mixture.ncomponents):
            beta                      = mixture.components[m, k].beta + epsilon
            beta                     /= numpy.sum(beta)
            mixture.components[m, k]  = Multinomial(beta)

def make_mult_mixture_model(training, ncomponents):
    """
    Return a new multinomial mixture strategy for evaluation.
    """

    log.info("building multinomial mixture model")

    from cargo.statistics.mixture     import EM_MixtureEstimator
    from cargo.statistics.multinomial import MultinomialEstimator
    from utexas.portfolio.models      import MultinomialMixtureActionModel

    model = \
        MultinomialMixtureActionModel(
            training,
            EM_MixtureEstimator(
                [[MultinomialEstimator()] * ncomponents] * len(training),
                ),
            )

    smooth_mult(model.mixture)

    return model

def main():
    """
    Main.
    """

    # get command line arguments
    import utexas.data

    from cargo.sql.alchemy import SQL_Engines
    from cargo.flags       import parse_given

    (samples_path, model_path) = parse_given()

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("cargo.statistics.mixture", level = "DEBUG")

    # load samples
    import cPickle as pickle

    with open(samples_path) as file:
        samples = pickle.load(file)

    # FIXME write the samples this way
    samples = dict((k, numpy.array(v)) for (k, v) in samples.items())

    # run inference and store model
    model = make_mult_mixture_model(samples, 16)

    with open(model_path, "w") as file:
        pickle.dump(model, file, -1)

