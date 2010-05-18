# vim: set fileencoding=UTF-8 :
"""
utexas/tools/portfolio/train.py

Solve a task using a portfolio.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.solve import main

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
        "Solver Execution Options",
        Flag(
            "-t",
            "--tasks",
            metavar = "FILE",
            help    = "load task names from FILE [%default]",
            ),
        Flag(
            "-s",
            "--solvers",
            metavar = "FILE",
            help    = "load solver names from FILE [%default]",
            ),
        Flag(
            "-c",
            "--cutoffs",
            metavar = "FILE",
            help    = "load cutoff times from FILE [%default]",
            ),
        )

# portfolio configuration information:
# - model name
# - model configuration
# - planner name
# - planner configuration

def select_tasks():
    sat_tasks        = chain(*[yield_tasks(p) for p in task_path_prefixes])
    tasks            = tuple(SAT_WorldTask(n, t) for (n, t) in enumerate(sat_tasks))

def yield_tasks(filter):
    """
    Yield the relevant tasks.
    """

    session       = ResearchSession()
    query_results =                                      \
        session                                          \
        .query(TaskDescription)                          \
        .filter(TaskDescription.name.startswith(filter))

    for description in query_results:
        yield description.task

def train_portfolio():

    seconds          = numpy.r_[2.0:5000.0:12j]
    durations        = [TimeDelta(seconds = s) for s in seconds]
    sat_2007_actions = build_actions(yield_sat2007_solvers(), durations)
    satzilla_actions = build_actions(yield_satzilla_solvers(), [TimeDelta(seconds = 5000.0)], firstn = len(sat_2007_actions))
    all_actions      = sat_2007_actions
#     all_actions      = sat_2007_actions + satzilla_actions
    world            = SAT_World(all_actions, tasks)
    nrestarts        = 16
    ntasks_test      = len(world.tasks) - ntasks_train

    log.info("using %i training and %i test tasks", ntasks_train, ntasks_test)

    (train_tasks, test_tasks) = random_subsets(world.tasks, (ntasks_train, ntasks_test))
    train_history             = world.act_all(train_tasks, sat_2007_actions, nrestarts)

    # build the strategy
    if isinstance(s_name, SAT_WorldAction):
        # fixed strategy
        strategy = FixedSelectionStrategy(s_name)
    elif s_name == "model":
        # model-based strategy; build the model
        if model_name == "dcm":
            model = make_dcm_mixture_model(world, train_history, K)
        elif model_name == "mult":
            model = make_mult_mixture_model(world, train_history, K)
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

@with_flags_parsed(
    usage = "usage: %prog [options] <portfolio.json>",
    )
def main(positional):
    """
    Main.
    """

    # basic flag handling
    flags = module_flags.given

    if flags.verbose:
        get_logger("utexas.tools.sat.run_solvers").setLevel(logging.NOTSET)
        get_logger("cargo.unix.accounting").setLevel(logging.DEBUG)
        get_logger("utexas.sat.solvers").setLevel(logging.DEBUG)

