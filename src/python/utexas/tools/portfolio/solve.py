# vim: set fileencoding=UTF-8 :
"""
utexas/tools/portfolio/solve.py

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
from numpy              import RandomState
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
            "-c",
            "--configuration",
            metavar = "FILE",
            help    = "load portfolio configuration from FILE [%default]",
            ),
        )

class SAT_PortfolioSolver(SAT_Solver):
    """
    Solve SAT instances with a portfolio.
    """

    def __init__(self, strategy):
        """
        Initialize.
        """

        self.strategy        = strategy
        self.max_invocations = 50

    def _solve(self, cutoff, input_path, seed = None):
        """
        Execute the solver and return its outcome, given an input path.
        """

        return \
            SAT_Result(
                self._solve_on(
                    cutoff,
                    SAT_WorldTask(input_path, input_path),
                    ),
                )

    def _solve_on(self, cutoff, task):
        """
        Evaluate on a specific task.
        """

        remaining = cutoff
        nleft     = self.max_invocations

        while remaining > TimeDelta() and nleft > 0:
            (action, pair)     = self._solve_once_on(task, remaining)
            (outcome, result)  = pair
            ntaken            += 1
            remaining         -= action.cost

            if result.satisfiable is not None:
                return result.satisfiable

        return None

    def _solve_once_on(self, strategy, task, remaining):
        """
        Evaluate once on a specific task.
        """

        # select an action
        action_generator = strategy.select(task, remaining)
        action           = action_generator.send(None)

        if action is None:
            return (None, None)

        # take it, and provide the outcome
        (outcome, result) = action.take(task)

        try:
            action_generator.send(outcome)
        except StopIteration:
            pass

        return (action, (outcome, result))

def build_portfolio():
    """
    Build portfolio.
    """

    all_actions = sat_2007_actions
    world       = SAT_World(all_actions, tasks)

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

        strategy = ModelPortfolio(world, model, planner)
    else:
        raise ValueError()

@with_flags_parsed(
    usage = "usage: %prog [options] <task>",
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

