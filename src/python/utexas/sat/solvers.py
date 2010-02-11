"""
utexas/sat/solvers.py

Run satisfiability solvers.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re
import numpy

from os import (
    fsync,
    getenv,
    )
from abc import abstractmethod
from copy import copy
from os.path import (
    join,
    splitext,
    basename,
    )
from tempfile import NamedTemporaryFile
from contextlib import closing
from cargo.io import openz
from cargo.log import get_logger
from cargo.unix.accounting import run_cpu_limited
from cargo.sugar import ABC
from cargo.flags import (
    Flag,
    Flags,
    )
from cargo.temporal import TimeDelta
from utexas.sat.cnf import write_sanitized_cnf

log = get_logger(__name__)

class SAT_Solver(ABC):
    """
    A solver for SAT.
    """

    def solve(self, cutoff, input_path, seed = None):
        """
        Execute the solver and return its outcome.

        @return (outcome, seconds_elapsed, seed)
        """

        # FIXME shouldn't necessarily assume CNF input
        # FIXME handle seed sanity ("does solver accept seed?" checking)

        with closing(openz(input_path)) as source:
            with NamedTemporaryFile(suffix = ".cnf") as temporary:
                log.info("writing %s from %s", temporary.name, input_path)

                write_sanitized_cnf(temporary, source)

                temporary.flush()

                fsync(temporary.fileno())

                return self._solve(cutoff, temporary.name, seed)

    @abstractmethod
    def _solve(self, cutoff, input_path, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        pass

class SATensteinSolver(SAT_Competition2007_Solver):
    """
    Some configuration of the SATenstein highly-parameterized solver.
    """

    class_flags = \
        Flags(
            "SATenstein Configuration",
            Flag(
                "--satenstein-path",
                default = "./satenstein",
                metavar = "PATH",
                help    = "solver is at PATH [%default]",
                ),
            )

    def __init__(self, parameters, flags = class_flags.given):
        """
        Initialize.
        """

        # build the argument list
        base_argv = ["-alg", "satenstein", "-r", "satcomp", "-cutoff", "-1"]
        more_argv = \
            sum(
                (["-%s" % n, "%f" % v] for (n, v) in parameters.items()),
                [],
                )

        # and construct the core solver interface
        self.flags       = self.class_flags.merged(flags)
        self.solver_core = \
            SAT_Competition2007_Solver(
                relative_path = self.flags.satenstein_path,
                seeded        = True,
                argv          = base_argv + more_argv,
                seed_args     = ["-seed"],
                input_args    = ["-i"],
                flags         = {"solvers_2007_path": "/"},
                )
        self._solve = self.solver_core._solve

class ArgoSAT_Solver(SAT_Solver):
    """
    Interface with ArgoSAT.
    """

    __sat_line_re   = re.compile(r"Model: \d+")
    __unsat_line_re = re.compile(r"Formula found unsatisfiable")

    seeded = True # FIXME ?

    class_flags = \
        Flags(
            "ArgoSAT Configuration",
            Flag(
                "--argosat-path",
                default = "./argosat",
                metavar = "PATH",
                help    = "solver is at PATH [%default]",
                ),
            )

    def __init__(self, argv = (), flags = class_flags.given):
        """
        Initialize.
        """

        # base
        SAT_Solver.__init__(self)

        # members
        self.argv  = tuple(argv)
        self.flags = self.class_flags.merged(flags)

    def _solve(self, cutoff, input_path, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        # run the solver
        command   = self.flags.argosat_path
        seed_argv = () if seed is None else ("--seed", str(seed))

        log.debug("running %s on input", command)

        (chunks, elapsed, exit_status) = \
            run_cpu_limited(
                (command, input_path) + tuple(self.argv) + seed_argv,
                cutoff,
                )

        # analyze its output
        if exit_status is None:
            return (None, elapsed)
        else:
            for line in "".join(ct for (_, ct) in chunks).split("\n"):
                if ArgoSAT_Solver.__sat_line_re.match(line):
                    return (True, elapsed)
                elif ArgoSAT_Solver.__unsat_line_re.match(line):
                    return (False, elapsed)

            return (None, elapsed)

