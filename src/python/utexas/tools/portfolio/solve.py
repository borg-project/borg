# vim: set fileencoding=UTF-8 :
"""
utexas/tools/portfolio/solve.py

Solve a task using a portfolio.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.solve import main

    raise SystemExit(main())

import logging
import numpy

from numpy.random               import RandomState
from cargo.log                  import get_logger
from cargo.flags                import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.temporal             import TimeDelta
from utexas.sat.solvers         import (
    SAT_Solver,
    SAT_Result,
    SAT_UncompressingSolver,
    SAT_PreprocessingSolver,
    get_named_solvers,
    )
from utexas.sat.preprocessors   import SatELitePreprocessor
from utexas.portfolio.models    import RandomActionModel
from utexas.portfolio.planners  import HardMyopicActionPlanner
from utexas.portfolio.sat_world import (
    SAT_WorldTask,
    SAT_WorldAction,
    )
from utexas.portfolio.strategies import ModelingSelectionStrategy

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
        Flag(
            "-s",
            "--seed",
            type    = int,
            default = 42,
            metavar = "INT",
            help    = "use INT to seed the internal PRNG [%default]",
            ),
        Flag(
            "-u",
            "--cutoff",
            type    = float,
            metavar = "FLOAT",
            help    = "run for at most ~FLOAT seconds [%default]",
            ),
        Flag(
            "-v",
            "--verbose",
            action  = "store_true",
            help    = "be noisier [%default]",
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

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Execute the solver and return its outcome, given an input path.
        """

        # get us a pseudorandom sequence
        if type(seed) is int:
            random = RandomState(seed)
        elif hasattr(seed, "rand"):
            random = seed
        else:
            raise ValueError("seed or PRNG required")

        # solve the instance
        (satisfiable, certificate) = \
            self._solve_on(
                SAT_WorldTask(input_path, input_path),
                cutoff,
                random,
                )

        return SAT_Result(satisfiable, certificate)

    def _solve_on(self, task, cutoff, random):
        """
        Evaluate on a specific task.
        """

        remaining = cutoff
        nleft     = self.max_invocations

        while (remaining is None or remaining > TimeDelta()) and nleft > 0:
            (action, pair)     = self._solve_once_on(task, remaining, random)
            (outcome, result)  = pair
            nleft             -= 1

            if remaining is not None:
                remaining -= action.cost

            if result.satisfiable is not None:
                return (result.satisfiable, result.certificate)

        return (None, None)

    def _solve_once_on(self, task, remaining, random):
        """
        Evaluate once on a specific task.
        """

        # select an action
        action_generator = self.strategy.select(task, remaining)
        action           = action_generator.send(None)

        if action is None:
            return (None, None)

        # take it, and provide the outcome
        (outcome, result) = action.take(task, random)

        try:
            action_generator.send(outcome)
        except StopIteration:
            pass

        return (action, (outcome, result))

# FIXME add a competition-compliant logging sink (ie, prepends "c ")

@with_flags_parsed(
    usage = "usage: %prog [options] <task>",
    )
def main((input_path,)):
    """
    Main.
    """

    # basic flag handling
    flags = module_flags.given

    if flags.verbose:
        get_logger("utexas.tools.sat.run_solvers").setLevel(logging.NOTSET)
        get_logger("cargo.unix.accounting").setLevel(logging.DEBUG)
        get_logger("utexas.sat.solvers").setLevel(logging.DEBUG)

    # load configuration
    # FIXME actually load configuration

    # solvers to use
    solver_names = [
        "sat/2009/clasp",
        "sat/2009/glucose",
#         "sat/2009/minisat_cumr_p",
        "sat/2009/mxc_09",
        "sat/2009/precosat",
        ]
    named_solvers = get_named_solvers()
    solvers       = map(named_solvers.__getitem__, solver_names)

    # instantiate the random strategy
    random   = RandomState(flags.seed)
    actions  = [SAT_WorldAction(s, TimeDelta(seconds = 4.0)) for s in solvers]
    strategy = \
        ModelingSelectionStrategy(
            RandomActionModel(random),
            HardMyopicActionPlanner(1.0),
            actions,
            )
    solver   = \
        SAT_UncompressingSolver(
            SAT_PreprocessingSolver(
                SatELitePreprocessor(),
                SAT_PortfolioSolver(strategy),
                ),
            )

    # run it
    if flags.cutoff is None:
        cutoff = None
    else:
        cutoff = TimeDelta(seconds = flags.cutoff)

    result = solver.solve(input_path, cutoff, seed = random)

    # tell the world
    if result.satisfiable is True:
        print "s SATISFIABLE"
        print "v %s" % " ".join(map(str, result.certificate))

        return 10
    elif result.satisfiable is False:
        print "s UNSATISFIABLE"

        return 20
    else:
        print "s UNKNOWN"

        return 0

