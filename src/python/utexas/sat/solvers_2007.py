"""
utexas/sat/solvers.py

Run satisfiability solvers.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re

from copy import copy
from os.path import join
from cargo.log import get_logger
from cargo.unix.accounting import run_cpu_limited
from cargo.flags import (
    Flag,
    Flags,
    )
from utexas.sat.solvers import SAT_Solver

log = get_logger(__name__)

SETTINGS_BY_NAME = {
    "adaptg2wsat+":     ("sat2007/bin/selected/adaptg2wsat+",     TRUE),
    "adaptg2wsat0":     ("sat2007/bin/selected/adaptg2wsat0",     TRUE),
    "adaptnovelty":     ("sat2007/bin/selected/adaptnovelty",     TRUE),
    "cmusat":           ("sat2007/bin/selected/cmusat",           TRUE),
    "dewSatz_1a":       ("sat2007/bin/selected/dewSatz_1a",       TRUE),
    "gnovelty+":        ("sat2007/bin/selected/gnovelty+",        TRUE),
    "kcnfs-2004":       ("sat2007/bin/selected/kcnfs-2004",       FALSE),
    "march_ks":         ("sat2007/bin/selected/march_ks",         FALSE),
    "minisat":          ("sat2007/bin/selected/minisat",          FALSE),
    "mxc":              ("sat2007/bin/selected/mxc",              FALSE),
    "picosat":          ("sat2007/bin/selected/picosat",          FALSE),
    "ranov":            ("sat2007/bin/selected/ranov",            TRUE),
    "rsat":             ("sat2007/bin/selected/rsat",             TRUE),
    "sapsrt":           ("sat2007/bin/selected/sapsrt",           TRUE),
    "sat7_bin":         ("sat2007/bin/selected/sat7_bin",         FALSE),
    "SatELite_release": ("sat2007/bin/selected/SatELite_release", FALSE),
    "satzillac":        ("sat2007/bin/selected/satzillac",        TRUE),
    "satzillaf":        ("sat2007/bin/selected/satzillaf",        TRUE),
    "satzillar":        ("sat2007/bin/selected/satzillar",        TRUE),
    "siege_v4":         ("sat2007/bin/selected/siege_v4",         TRUE),
    "tts-4-0":          ("sat2007/bin/selected/tts-4-0",          TRUE),
    }

class SAT_Competition2007_Solver(SAT_Solver):
    """
    A solver for SAT that uses the circa-2007 competition interface.
    """

    class_flags = \
        Flags(
            "SAT 2007 Solvers Configuration",
            Flag(
                "--solvers-2007-path",
                default = ".",
                metavar = "PATH",
                help    = "find 2007 solvers under PATH [%default]",
                ),
            )

    def __init__(
        self,
        relative_path,
        seeded     = True,
        argv       = [],
        seed_args  = [],
        input_args = [],
        flags      = class_flags.given,
        ):
        """
        Initialize this solver.
        """

        # base
        SAT_Solver.__init__(self)

        # members
        self.relative_path = relative_path
        self.seeded        = bool(seeded)
        self.argv          = copy(argv)
        self.seed_args     = copy(seed_args)
        self.input_args    = copy(input_args)
        self.flags         = self.class_flags.merged(flags)

    def _solve(self, cutoff, input_path, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        # run the solver
        command   = join(self.flags.solvers_2007_path, self.relative_path)
        seed_argv = [] if seed is None else self.seed_args + [str(seed)]
        input_argv = self.input_args + [input_path]

        log.debug("running %s on input", command)

        (chunks, elapsed, exit_status) = \
            run_cpu_limited(
                [command] + input_argv + seed_argv + self.argv,
                cutoff,
                )

        if exit_status == 10:
            outcome = True
        elif exit_status == 20:
            outcome = False
        else:
            outcome = None

        return (outcome, elapsed)

    @staticmethod
    def by_name(
        name,
        flags = class_flags.given,
        ):
        """
        Construct a solver by name.
        """

        return SAT_Competition2007_Solver(SETTINGS_BY_NAME[name], flags)

