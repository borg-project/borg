"""
Calculate a machine speed score.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                   import call
    from utexas.tools.calibrate import main

    call(main)

from plac      import annotations
from cargo.log import get_logger

log = get_logger(__name__, default_level = "NOTSET")

def get_solver(named_solvers, name):
    """
    Build a complete solver.
    """

    from utexas.sat.solvers import (
        SAT_SanitizingSolver,
        SAT_UncompressingSolver,
        )

    return SAT_UncompressingSolver(SAT_SanitizingSolver(named_solvers[name]), name)

@annotations(
    calibration = ("configuration file", "positional"),
    repeats     = ("run count"         , "option"     , "r", int),
    )
def main(calibration, repeats = 4):
    """
    Application body.
    """

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("cargo.unix.accounting", level = "DETAIL")

    # build the solvers
    from utexas.sat.solvers import get_named_solvers

    named_solvers = get_named_solvers()

    # build the calibration runs
    import json

    with open(calibration) as file:
        runs = json.load(file)

    # and execute them
    from datetime       import timedelta
    from cargo.io       import expandpath
    from cargo.temporal import seconds

    outer_total = timedelta()

    for (solver_name, path, seed) in runs:
        solver      = get_solver(named_solvers, solver_name)
        full_path   = expandpath(path)
        inner_total = timedelta()

        for i in xrange(repeats):
            result       = solver.solve(full_path, timedelta(seconds = 1e6), seed)
            inner_total += result.run.usage_elapsed

        average      = inner_total / repeats
        outer_total += average

        log.detail("run average is %f", seconds(average))

    score = outer_total / len(runs)

    log.note("machine performance score is %f", seconds(score))

