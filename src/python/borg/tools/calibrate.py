"""
Calculate a machine speed score.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.calibrate import main

    raise SystemExit(main())

from cargo.log   import get_logger
from cargo.flags import (
    Flag,
    Flags,
    with_flags_parsed,
    )

log          = get_logger(__name__, default_level = "NOTSET")
module_flags = \
    Flags(
        "Script Options",
        Flag(
            "-r",
            "--repeats",
            type    = int,
            default = 4,
            metavar = "INT",
            help    = "run each solver INT times [%default]",
            ),
        Flag(
            "-c",
            "--calibration",
            metavar = "FILE",
            help    = "use calibration configuration FILE [%default]",
            ),
        )

def get_solver(named_solvers, name):
    """
    Build a complete solver.
    """

    from utexas.sat.solvers import (
        SAT_SanitizingSolver,
        SAT_UncompressingSolver,
        )

    return SAT_UncompressingSolver(SAT_SanitizingSolver(named_solvers[name]), name)

def main():
    """
    Application body.
    """

    # get command line arguments
    import utexas.sat.solvers

    from cargo.flags import parse_given

    parse_given()

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("cargo.unix.accounting", level = "DETAIL")

    # build the solvers
    from utexas.sat.solvers import get_named_solvers

    named_solvers = get_named_solvers()

    # build the calibration runs
    import json

    with open(module_flags.given.calibration) as file:
        runs = json.load(file)

    # and execute them
    from cargo.io       import expandpath
    from cargo.temporal import TimeDelta

    nrepeats    = module_flags.given.repeats
    outer_total = TimeDelta()

    for (solver_name, path, seed) in runs:
        solver      = get_solver(named_solvers, solver_name)
        full_path   = expandpath(path)
        inner_total = TimeDelta()

        for i in xrange(nrepeats):
            result       = solver.solve(full_path, TimeDelta(seconds = 1e6), seed)
            inner_total += result.run.usage_elapsed

        average      = inner_total / nrepeats
        outer_total += average

        log.detail("run average is %f", TimeDelta.from_timedelta(average).as_s)

    score = outer_total / len(runs)

    log.note("machine performance score is %f", TimeDelta.from_timedelta(score).as_s)

