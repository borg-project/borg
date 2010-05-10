# vim: set fileencoding=UTF-8 :
"""
Solve a task using a portfolio.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.solve import main

    raise SystemExit(main())

import utexas.sat.preprocessors

from logging                    import Formatter
from cargo.log                  import get_logger
from cargo.flags                import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from utexas.sat.solvers         import (
    SAT_Solver,
    SAT_BareResult,
    )
from utexas.portfolio.sat_world import (
    SAT_WorldTask,
    SAT_WorldAction,
    )

log = get_logger(__name__, default_level = "NOTE")

module_flags = \
    Flags(
        "Solver Execution Options",
        Flag(
            "-s",
            "--seed",
            type    = int,
            default = 43,
            metavar = "INT",
            help    = "use INT to seed the internal PRNG [%default]",
            ),
        Flag(
            "-c",
            "--calibration",
            type    = float,
            default = 5.59,
            metavar = "FLOAT",
            help    = "assume machine speed FLOAT [%default]",
            ),
        Flag(
            "-m",
            "--model",
            metavar = "PATH",
            help    = "read model from PATH [%default]",
            ),
        Flag(
            "-v",
            "--verbose",
            action  = "store_true",
            help    = "be noisier [%default]",
            ),
        )

class SAT_PortfolioResult(SAT_BareResult):
    """
    Result of a portfolio solver.
    """

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

        from numpy.random import RandomState

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

        return SAT_PortfolioResult(satisfiable, certificate)

    def _solve_on(self, task, cutoff, random):
        """
        Evaluate on a specific task.
        """

        from cargo.temporal import TimeDelta

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

class CompetitionFormatter(Formatter):
    """
    A concise log formatter for output during competition.
    """

    def __init__(self):
        """
        Initialize.
        """

        Formatter.__init__(self, "%(levelname)s: %(message)s", "%y%m%d%H%M%S")

    def format(self, record):
        """
        Format the log record.
        """

        raw = Formatter.format(self, record)

        def yield_lines():
            """
            Yield comment-prefixed lines.
            """

            lines  = raw.splitlines()
            indent = "c " + " " * (len(record.levelname) + 2)

            yield "c " + lines[0]

            for line in lines[1:]:
                yield indent + line

        return "\n".join(yield_lines())

@with_flags_parsed(
    usage = "usage: %prog [options] <task>",
    )
def main((input_path,)):
    """
    Main.
    """

    # set up competition logging
    import sys
    import logging

    from logging   import StreamHandler
    from cargo.log import enable_default_logging

    enable_default_logging(add_handlers = False)

    handler = StreamHandler(sys.stdout)

    handler.setFormatter(CompetitionFormatter())
    handler.setLevel(logging.NOTSET)

    logging.root.addHandler(handler)

    # basic flag handling
    flags = module_flags.given

    if flags.verbose:
        get_logger("cargo.unix.accounting",        level = "DETAIL")
        get_logger("utexas.sat.solvers",           level = "DETAIL")
        get_logger("utexas.sat.preprocessors",     level = "DETAIL")
        get_logger("utexas.tools.sat.run_solvers", level = "NOTSET")
        get_logger("utexas.portfolio.models",      level = "NOTSET")

    # solvers to use
    from utexas.sat.solvers import (
        SAT_Solver,
        SAT_BareResult,
        SAT_UncompressingSolver,
        SAT_PreprocessingSolver,
        get_named_solvers,
        )

    named_solvers = get_named_solvers()

    # instantiate the strategy
    from itertools                   import product
    from numpy                       import r_
    from numpy.random                import RandomState
    from cargo.temporal              import TimeDelta
    from utexas.sat.preprocessors    import SatELitePreprocessor
    from utexas.portfolio.models     import RandomActionModel
    from utexas.portfolio.planners   import HardMyopicActionPlanner
    from utexas.portfolio.strategies import ModelingSelectionStrategy

    random = RandomState(flags.seed)

    if flags.model is not None:
        # configurable model-based portfolio
        import cPickle as pickle

        with open(flags.model) as file:
            model = pickle.load(file)

        r              = flags.calibration / 2.1 # hardcoded rhavan score
        map_action     = lambda (s, c): SAT_WorldAction(named_solvers[s], TimeDelta(seconds = c.as_s * r))
        actions        = map(map_action, model._actions)
        model._actions = actions
        planner        = HardMyopicActionPlanner(1.0 - 2e-3)
    else:
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

    strategy = \
        ModelingSelectionStrategy(
            model,
            planner,
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
    result = solver.solve(input_path, TimeDelta(seconds = 1e6), seed = random)

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

