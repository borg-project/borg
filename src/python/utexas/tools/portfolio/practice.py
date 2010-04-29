# vim: set fileencoding=UTF-8 :
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.practice import main

    raise SystemExit(main())

import re
import json
import logging
import numpy

from os.path            import (
    join,
    dirname,
    )
from copy               import copy
from tempfile           import NamedTemporaryFile
from cargo.io           import expandpath
from cargo.log          import get_logger
from cargo.json         import follows
from cargo.flags        import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.temporal     import TimeDelta
from utexas.sat.solvers import get_named_solvers

log = get_logger(__name__, level = logging.NOTE)

module_flags = \
    Flags(
        "Script Options",
        Flag(
            "-t",
            "--training",
            default = "training.json",
            metavar = "FILE",
            help    = "load training data from FILE [%default]",
            ),
        Flag(
            "-o",
            "--output",
            default = "portfolio.json",
            metavar = "FILE",
            help    = "write configuration to FILE [%default]",
            ),
        )

# portfolio configuration information:
# - model name
# - model configuration
# - planner name
# - planner configuration

def smooth_mult(mixture):
    """
    Apply a smoothing term to the multinomial mixture components.
    """

    log.info("heuristically smoothing mult mixture")

    epsilon = 1e-3

    for m in xrange(mixture.ndomains):
        for k in xrange(mixture.ncomponents):
            beta                      = mixture.components[m, k].beta + epsilon
            beta                     /= numpy.sum(beta)
            mixture.components[m, k]  = Multinomial(beta)

def make_mult_mixture_model(world, training, ncomponents):
    """
    Return a new multinomial mixture strategy for evaluation.
    """

    log.info("building multinomial mixture model")

    model  = \
        MultinomialMixtureActionModel(
            world,
            training,
            ExpectationMaximizationMixtureEstimator(
                [[MultinomialEstimator()] * ncomponents] * world.nactions,
                ),
            )

    smooth_mult(model.mixture)

    return model

def train_portfolio(tasks, solvers, cutoffs):
    """
    Return trained portfolio.
    """

    model = make_mult_mixture_model(world, train_history, 4)

"""
    # build the strategy
    if isinstance(s_name, SAT_WorldAction):
        # fixed strategy
        strategy = FixedSelectionStrategy(s_name)
    elif s_name == "model":
        # model-based strategy; build the model
        if model_name == "dcm":
            model = make_dcm_mixture_model(world, train_history, K)
        elif model_name == "random":
            model = RandomActionModel(world)
        else:
            raise ValueError()

        # build the planner
        if planner_name == "hard":
            planner = HardMyopicActionPlanner(world, discount)
        elif planner_name == "soft":
            planner = SoftMyopicActionPlanner(world, discount)
        else:
            raise ValueError()

        strategy = ModelingSelectionStrategy(world, model, planner)
    else:
        raise ValueError()
"""

def get_samples_for(session, solver_name, cutoff, task):
    """
    Build a list of samples.
    """

    from sqlalchemy import (
        and_,
        func,
        select,
        )
    from utexas.data import (
        SAT_AttemptRow              as SA,
        SAT_RunAttemptRow           as SRA,
        SAT_PreprocessingAttemptRow as SPA,
        )

    sa0   = SA.__table__.alias()
    spa0  = SPA.__table__
    query = \
        select(
            [SA.cost, SRA.satisfiable],
            and_(
                SA.uuid                  == SRA.__table__.c.uuid,
                SRA.solver_name          == solver_name,
                SRA.budget               >= cutoff,
                SRA.uuid                 == spa0.c.inner_attempt_uuid,
                spa0.c.preprocessor_name == "SatELite",
                spa0.c.uuid              == sa0.c.uuid,
                sa0.c.task_uuid          == task.uuid,
                )
            )

    nruns   = 0
    nsolved = 0

    for (c, s) in session.execute(query):
        nruns += 1

        if c <= cutoff and s:
            nsolved += 1

    if nruns > 0:
        print solver_name, cutoff, task.uuid, float(nsolved) / nruns
    else:
        print solver_name, cutoff, task.uuid, "*"

    return numpy.array([nsolved, nruns - nsolved], numpy.uint)

def get_samples(session):
    """
    Build a list of samples.
    """

    # get the solver names
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

    # get the tasks
    from utexas.data import TaskNameRow

    query_results =                                                              \
        session                                                                  \
        .query(TaskNameRow)                                                      \
        .filter(TaskNameRow.name.startswith("sat/competition_2009/industrial/"))

    log.debug("found %i tasks", query_results.count())

    tasks = [n.task for n in query_results]

    # get the cutoffs
    from numpy import r_

    cutoffs = [TimeDelta(seconds = c) for c in r_[10.0:800.0:6j]]

    # get the samples
    from itertools import product
    from functools import partial

    samples = {}

    for k in product(solver_names, cutoffs):
        (solver_name, cutoff) = k
        f                     = partial(get_samples_for, session, solver_name, cutoff)
        samples[k]            = map(f, tasks)

    return samples

def main():
    """
    Main.
    """

    # get command line arguments
    import utexas.data

    from cargo.sql.alchemy import SQL_Engines
    from cargo.flags       import parse_given

    (samples_path,) = parse_given()

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    # practice
    with SQL_Engines.default:
        # grab the list of samples
        from contextlib  import closing
        from utexas.data import (
            ResearchSession,
            research_connect,
            )

        ResearchSession.configure(bind = research_connect())

        with closing(ResearchSession()) as session:
            samples = get_samples(session)

        # and store them
        import cPickle as pickle

        with open(samples_path, "w") as file:
            pickle.dump(samples, file, -1)

