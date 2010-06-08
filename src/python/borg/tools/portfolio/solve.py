"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.portfolio.solve import main

    raise SystemExit(main())

import borg.solvers

from logging     import Formatter
from cargo.log   import get_logger
from cargo.flags import (
    Flag,
    Flags,
    with_flags_parsed,
    )

log = get_logger(__name__, default_level = "NOTE")

module_flags = \
    Flags(
        "Solver Execution Options",
        Flag(
            "-c",
            "--calibration",
            type    = float,
            default = 5.59,
            metavar = "FLOAT",
            help    = "assume machine speed FLOAT [%default]",
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

def enable_output():
    """
    Set up competition-compliant output.
    """

    import sys
    import logging

    from logging   import StreamHandler
    from cargo.log import enable_default_logging

    enable_default_logging(add_handlers = False)

    handler = StreamHandler(sys.stdout)

    handler.setFormatter(CompetitionFormatter())
    handler.setLevel(logging.NOTSET)

    logging.root.addHandler(handler)

@with_flags_parsed(
    usage = "usage: %prog [options] <solver.pickle> <task> <seed>",
    )
def main((solver_path, input_path, seed_string)):
    """
    Main.
    """

    # allow some logging output
    enable_output()

    # basic flag handling
    flags = module_flags.given

    if flags.verbose:
        get_logger("cargo.unix.accounting", level = "DETAIL")
        get_logger("borg.portfolio.models", level = "NOTSET")

    # build our PRNG
    from numpy.random import RandomState

    random = RandomState(int(seed_string))

    # instantiate the strategy
    import cPickle as pickle

    with open(solver_path) as file:
        solver = pickle.load(file)

    # build the solver environment
    from borg.solvers import (
        Environment,
        get_named_solvers,
        )

    environment = \
        Environment(
            named_solvers = get_named_solvers(),
            time_ratio    = 2.2 / flags.calibration,
            )

    # solve
    from cargo.temporal import TimeDelta
    from borg.tasks     import FileTask
    from borg.solvers   import UncompressingSolver

    task        = FileTask(input_path)
    full_solver = UncompressingSolver(solver)
    attempt     = full_solver.solve(task, TimeDelta(seconds = 1e6), random, environment)
    answer      = attempt.answer

    # tell the world
    # FIXME should be domain-specific
    if answer is None:
        print "s UNKNOWN"

        return 0
    elif answer.satisfiable:
        print "s SATISFIABLE"
        print "v %s" % " ".join(map(str, answer.certificate))

        return 10
    else:
        print "s UNSATISFIABLE"

        return 20

