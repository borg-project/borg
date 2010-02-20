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

class SAT_CompetitionSolver(SAT_Solver):
    """
    A solver for SAT that uses the circa-2009 competition interface.
    """

    __sat_line_re   = re.compile("s SATISFIABLE")
    __unsat_line_re = re.compile("s UNSATISFIABLE")

    class_flags = \
        Flags(
            "SAT Competition Solvers Configuration",
            Flag(
                "--competition-solvers-path",
                default = ".",
                metavar = "PATH",
                help    = "find SAT competition solvers under PATH [%default]",
                ),
            )

    def __init__(
        self,
        command,
        memlimit = None,
        flags    = class_flags.given,
        ):
        """
        Initialize this solver.

        @param memlimit: The memory limit reported to the solver.
        """

        # base
        SAT_Solver.__init__(self)

        # members
        self.command  = copy(command)
        self.memlimit = memlimit
        self.flags    = self.class_flags.merged(flags)

    def _solve(self, cutoff, input_path, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.

        Comments quote the competition specification.
        """

        def expand(strings, variable, value):
            """
            Expand occurences of variable in string with value.
            """

            return [s.replace(variable, str(value)) for s in strings]

        # expand variables in an element of a solver command string
        expanded    = self.command
        environment = {}

        # BENCHNAME: the name of the file (with both path and extension)
        # containing the instance to solve
        expanded = expand(expanded, "BENCHNAME", input_path)

        # BENCHNAMENOEXT: name of the file with path but without extension),
        (without, _) = splitext(input_path)
        expanded     = expand(expanded, "BENCHNAMENOEXT", without)

        # BENCHNAMENOPATH: name of the file without path but with extension
        base     = basename(input_path)
        expanded = expand(expanded, "BENCHNAMENOPATH", base)

        # BENCHNAMENOPATHNOEXT: name of the file without path nor extension
        base_without = basename(without)
        expanded     = expand(expanded, "BENCHNAMENOPATHNOEXT", base_without)

        # RANDOMSEED: a random seed which is a number between 0 and 4294967295
        if seed is not None:
            expanded = expand(expanded, "RANDOMSEED", seed)
        elif any("RANDOMSEED" in s for s in self.command):
            raise ValueError("no seed provided for seeded solver")

        # TIMELIMIT (or TIMEOUT): the total CPU time (in seconds) that the
        # solver may use before being killed
        cutoff_s                  = TimeDelta.from_timedelta(cutoff).as_s
        expanded                  = expand(expanded, "TIMELIMIT", cutoff_s)
        expanded                  = expand(expanded, "TIMEOUT", cutoff_s)
        environment["TIMELIMIT"]  = "%.1f" % cutoff_s
        environment["TIMEOUT"]    = environment["TIMELIMIT"]

        # SATTIMEOUT is a synonym for TIMELIMIT used by pre-2009 solvers
        environment["SATTIMEOUT"] = environment["TIMELIMIT"]

        # only report memlimit if requested to do so
        if self.memlimit is not None:
            # MEMLIMIT: the total amount of memory (in MiB) that the solver may use
            memlimit                = 1024
            expanded                = expand(expanded, "MEMLIMIT", memlimit)
            environment["MEMLIMIT"] = memlimit

            # SATRAM is a synonym for MEMLIMIT potentially used by pre-2009 solvers
            environment["SATRAM"]   = environment["MEMLIMIT"]

        # TMPDIR: the only directory where the solver is allowed to read/write
        # temporary files
        tmpdir                = getenv("TMPDIR", "/tmp")
        expanded              = expand(expanded, "TMPDIR", tmpdir)
        environment["TMPDIR"] = tmpdir

        # DIR: the directory where the solver files will be stored
        home                       = self.flags.competition_solvers_path
        expanded                   = expand(expanded, "DIR", home)
        expanded                   = expand(expanded, "HOME", home)
        environment["SOLVERSHOME"] = home

        # run the solver
        log.debug("running %s", expanded)

        (chunks, elapsed, exit_status) = run_cpu_limited(expanded, cutoff, environment)
        supplementary                  = {
            "output"   : "".join(c for (_, c) in chunks),
            "exit_code": exit_status,
            }

        # analyze its output
        for line in "".join(c for (t, c) in chunks if t <= cutoff).split("\n"):
            if SAT_CompetitionSolver.__sat_line_re.match(line):
                return (True, elapsed, supplementary)
            elif SAT_CompetitionSolver.__unsat_line_re.match(line):
                return (False, elapsed, supplementary)

        return (None, elapsed, supplementary)

    @property
    def seeded(self):
        """
        Is the solver seeded?
        """

        return any("RANDOMSEED" in s for s in self.command)

    @staticmethod
    def by_name(
        name,
        flags = class_flags.given,
        ):
        """
        Build a solver by name.
        """

        return SAT_CompetitionSolver(COMMANDS_BY_NAME[name], flags)

class SATensteinSolver(SAT_CompetitionSolver):
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

