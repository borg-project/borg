"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac             import call
    from borg.tools.solve import main

    call(main)

from logging   import Formatter
from plac      import annotations
from cargo.log import get_logger

log = get_logger(__name__, default_level = "NOTE")

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

@annotations(
    solver_path  = ("path to solver pickle", )           ,
    input_path   = ("path to instance"     , )           ,
    seed         = ("PRNG seed"            , "positional", None, int)  ,
    calibration  = ("speed factor"         , "option"    , "c" , float),
    preprocessor = ("preprocessor name"    , "option"    , "p"),
    quiet        = ("be less noisy"        , "flag"      , "q")
    )
def main(
    solver_path,
    input_path,
    seed         = 42,
    calibration  = 5.59,
    preprocessor = None,
    quiet        = False,
    ):
    """
    Solve a problem instance.
    """

    # allow some logging output
    enable_output()

    # configure logging
    if not quiet:
        get_logger("cargo.unix.accounting"  , level = "DETAIL")
        get_logger("borg.portfolio.models"  , level = "NOTSET")
        get_logger("borg.portfolio.planners", level = "NOTSET")
        get_logger("borg.solvers.satelite"  , level = "INFO")
        get_logger("borg.solvers.portfolio" , level = "INFO")

    # build our PRNG
    from numpy.random import RandomState

    random = RandomState(seed)

    # instantiate the strategy
    import cPickle as pickle

    with open(solver_path) as file:
        solver = pickle.load(file)

    # build the solver environment
    from borg.solvers           import (
        Environment,
        get_named_solvers,
        )

    environment = \
        Environment(
            named_solvers = get_named_solvers(),
            time_ratio    = calibration / 2.2,
            )

    # solve
    from datetime       import timedelta
    from borg.tasks     import FileTask
    from borg.solvers   import (
        LookupPreprocessor,
        UncompressingSolver,
        PreprocessingSolver,
        )

    if preprocessor is None:
        secondary = solver
    else:
        secondary = PreprocessingSolver(LookupPreprocessor(preprocessor), solver)

    primary = UncompressingSolver(secondary)
    task    = FileTask(input_path)
    attempt = primary.solve(task, timedelta(seconds = 2e6), random, environment)
    answer  = attempt.answer

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

