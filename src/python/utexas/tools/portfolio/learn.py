# vim: set fileencoding=UTF-8 :
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.learn import main

    raise SystemExit(main())

import numpy

from cargo.log   import get_logger
from cargo.flags import (
    Flag,
    Flags,
    )

log          = get_logger(__name__)
module_flags = \
    Flags(
        "Script Options",
        Flag(
            "-m",
            "--model-type",
            choices = ["multinomial", "dcm", "random"],
            default = "dcm",
            metavar = "TYPE",
            help    = "learn a TYPE model [%default]",
            ),
        Flag(
            "-r",
            "--restarts",
            type    = int,
            default = 2,
            metavar = "INT",
            help    = "make INT inference restarts [%default]",
            ),
        Flag(
            "-k",
            "--components",
            type    = int,
            default = 16,
            metavar = "INT",
            help    = "use INT latent classes [%default]",
            ),
        )

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

def make_mult_mixture_model(training, ncomponents, nrestarts):
    """
    Return a new multinomial mixture strategy for evaluation.
    """

    log.info("building multinomial mixture model")

    from cargo.statistics.mixture     import (
        RestartedEstimator,
        EM_MixtureEstimator,
        )
    from cargo.statistics.multinomial import MultinomialEstimator
    from utexas.portfolio.models      import MultinomialMixtureActionModel

    model = \
        MultinomialMixtureActionModel(
            training,
            RestartedEstimator(
                EM_MixtureEstimator(
                    [[MultinomialEstimator()] * ncomponents] * len(training),
                    ),
                nrestarts = nrestarts,
                )
            )

    smooth_mult(model.mixture)

    return model

def smooth_dcm(mixture):
    """
    Apply a smoothing term to the DCM mixture components.
    """

    log.info("heuristically smoothing DCM mixture")

    from cargo.statistics.dcm import DirichletCompoundMultinomial

    # find the smallest non-zero dimension
    smallest = numpy.inf
    epsilon  = 1e-6

    for components in mixture.components:
        for component in components:
            for v in component.alpha:
                if v < smallest and v > epsilon:
                    smallest = v

    if numpy.isinf(smallest):
        smallest = epsilon

    log.debug("smallest nonzero value is %f", smallest)

    for m in xrange(mixture.ndomains):
        for k in xrange(mixture.ncomponents):
            alpha                    = mixture.components[m, k].alpha
            mixture.components[m, k] = DirichletCompoundMultinomial(alpha + smallest * 1e-2)

def make_dcm_mixture_model(training, ncomponents, nrestarts):
    """
    Return a new DCM mixture model.
    """

    log.info("building DCM mixture model")

    from cargo.statistics.dcm     import DCM_Estimator
    from cargo.statistics.mixture import (
        RestartedEstimator,
        EM_MixtureEstimator,
        )
    from utexas.portfolio.models  import DCM_MixtureActionModel

    model  = \
        DCM_MixtureActionModel(
            training,
            RestartedEstimator(
                EM_MixtureEstimator(
                    [[DCM_Estimator()] * ncomponents] * len(training),
                    ),
                nrestarts = nrestarts,
                ),
            )

    smooth_dcm(model.mixture)

    return model

def make_random_model():
    # hardcoded random portfolio
    solver_names = [
        "sat/2009/CirCUs",
        "sat/2009/clasp",
        "sat/2009/glucose",
        "sat/2009/LySAT_i",
        "sat/2009/minisat_09z",
        "sat/2009/minisat_cumr_p",
        "sat/2009/mxc_09",
        "sat/2009/precosat",
        "sat/2009/rsat_09",
        "sat/2009/SApperloT",
        ]
    solvers = map(named_solvers.__getitem__, solver_names)
    cutoffs = [TimeDelta(seconds = c) for c in r_[10.0:800.0:6j]]
    actions = [SAT_WorldAction(*a) for a in product(solvers, cutoffs)]
    model   = RandomActionModel(random)
    planner = HardMyopicActionPlanner(1.0)

def main():
    """
    Main.
    """

    # get command line arguments
    import utexas.data

    from cargo.sql.alchemy import SQL_Engines
    from cargo.flags       import parse_given

    (samples_path, solver_path) = parse_given()

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("cargo.statistics.mixture", level = "DETAIL")

    # load samples
    import cPickle as pickle

    with open(samples_path) as file:
        samples = pickle.load(file)

    # FIXME write the samples this way
    samples = dict((k, numpy.array(v)) for (k, v) in samples.items())

    # run inference and build model
    model_type  = module_flags.given.model_type
    ncomponents = module_flags.given.components
    nrestarts   = module_flags.given.restarts

    if model_type == "dcm":
        model = make_dcm_mixture_model(samples, ncomponents, nrestarts)
    elif model_type == "multinomial":
        model = make_mult_mixture_model(samples, ncomponents, nrestarts)
    elif model_type == "random":
        model = make_random_model(samples)

    with open(model_path, "w") as file:
        pickle.dump(model, file, -1)

    # build the entire solver
    r              = flags.calibration / 2.1 # hardcoded rhavan score
    map_action     = lambda (s, c): SAT_WorldAction(named_solvers[s], TimeDelta(seconds = c.as_s * r))
    actions        = map(map_action, model._actions)
    model._actions = actions
    planner        = HardMyopicActionPlanner(1.0 - 2e-3)
    strategy       = \
        ModelingSelectionStrategy(
            model,
            planner,
            actions,
            )
    solver         = \
        SAT_UncompressingSolver(
            SAT_PreprocessingSolver(
                SatELitePreprocessor(),
                SAT_PortfolioSolver(strategy),
                ),
            )

    # write it to disk
    with open(solver_path, "w") as file:
        pickle.dump(file, solver)

