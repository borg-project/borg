"""
utexas/quest/acridid/argo_plot.py

General support routines.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.io import files_under
from cargo.ai.sat.solvers import ArgoSAT_Solver
from cargo.log import get_logger
from cargo.flags import (
    Flag,
    FlagSet,
    with_flags_parsed,
    )

log = get_logger(__name__, level = None)

class ModuleFlags(FlagSet):
    """
    Flags that apply to this module.
    """

    flag_set_title = "Script Configuration"

    benchmark_root_flag = \
        Flag(
            "--benchmark-root",
            default = ".",
            metavar = "PATH",
            help = "run on CNF instances under PATH [%default]",
            )

flags = ModuleFlags.given

@with_flags_parsed()
def main(positional):
    """
    Application body.
    """

    solver = ArgoSAT_Solver()

    for file in files_under(flags.benchmark_root, "*.cnf"):
        (outcome, elapsed, _) = solver.solve(4.0, file)

        log.info("solver returned %s after %.2fs", outcome, elapsed)

if __name__ == '__main__':
    main()

