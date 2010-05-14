# vim: set fileencoding=UTF-8 :
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.solve import main

    raise SystemExit(main())

from logging                    import Formatter
from cargo.log                  import get_logger
from cargo.flags                import (
    Flag,
    Flags,
    with_flags_parsed,
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
        get_named_solvers,
        )

    named_solvers = get_named_solvers()

    # instantiate the strategy
    from numpy.random import RandomState

    random = RandomState(flags.seed)

    if flags.model is not None:
        # configurable model-based portfolio
        import cPickle as pickle

        with open(flags.model) as file:
            solver = pickle.load(file)

    # run it
    from cargo.temporal   import TimeDelta
    from utexas.sat.tasks import SAT_FileTask

    task   = SAT_FileTask(input_path)
    result = solver.solve(task, TimeDelta(seconds = 1e6), random, environment)

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

