"""
cargo/sat/solvers.py

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
from utexas.quest.acridid.sat.cnf import write_sanitized_cnf

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
                # FIXME remove -i
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

class SAT_Competition2009_Solver(SAT_Solver):
    """
    A solver for SAT that uses the circa-2009 competition interface.
    """

    __sat_line_re   = re.compile("s SATISFIABLE")
    __unsat_line_re = re.compile("s UNSATISFIABLE")

    class_flags = \
        Flags(
            "SAT 2009 Solvers Configuration",
            Flag(
                "--solvers-2009-path",
                default = ".",
                metavar = "PATH",
                help    = "find 2009 solvers under PATH [%default]",
                ),
            )

    def __init__(
        self,
        command,
        flags = class_flags.given,
        ):
        """
        Initialize this solver.
        """

        # base
        SAT_Solver.__init__(self)

        # members
        self.command = copy(command)
        self.flags   = self.class_flags.merged(flags)

    def _solve(self, cutoff, input_path, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        def expand(strings, variable, value):
            """
            Expand occurences of variable in string with value.
            """

            return [s.replace(variable, str(value)) for s in strings]

        # expand variables in an element of a SAT 2009 command string
        expanded    = self.command
        environment = {}

        # - BENCHNAME will be replaced by the name of the file (with both path
        #   and extension) containing the instance to solve. Obviously, the
        #   solver must use this parameter or one of the following variants:
        expanded = expand(expanded, "BENCHNAME", input_path)

        # - BENCHNAMENOEXT  (name of the file with path but without extension),
        (without, _) = splitext(input_path)
        expanded     = expand(expanded, "BENCHNAMENOEXT", without)

        # - BENCHNAMENOPATH  (name of the file without path but with extension)
        base     = basename(input_path)
        expanded = expand(expanded, "BENCHNAMENOPATH", base)

        # - BENCHNAMENOPATHNOEXT (name of the file without path nor extension)
        base_without = basename(without)
        expanded     = expand(expanded, "BENCHNAMENOPATHNOEXT", base_without)

        # - RANDOMSEED will be replaced by a random seed which is a number
        #   between 0 and 4294967295. This parameter MUST be used to initialize
        #   the random number generator when the solver uses random numbers. It
        #   is recorded by the evaluation environment and will allow to run the
        #   program on a given instance under the same conditions if necessary.
        if seed is not None:
            expanded = expand(expanded, "RANDOMSEED", seed)
        elif any("RANDOMSEED" in s for s in self.command):
            raise ValueError("no seed provided for seeded solver")

        # - TIMELIMIT (or TIMEOUT) represents the total CPU time (in seconds)
        #   that the solver may use before being killed. May be used to adapt the
        #   solver strategy.
        cutoff_s                 = TimeDelta.from_timedelta(cutoff).as_s
        expanded                 = expand(expanded, "TIMELIMIT", cutoff_s)
        expanded                 = expand(expanded, "TIMEOUT", cutoff_s)
        environment["TIMELIMIT"] = "%.1f" % cutoff_s
        environment["TIMEOUT"]   = environment["TIMELIMIT"]

        # - MEMLIMIT represents the total amount of memory (in MiB) that the
        #   solver may use before being killed. May be used to adapt the solver
        #   strategy.   
        memlimit                = 1024
        expanded                = expand(expanded, "MEMLIMIT", memlimit)
        environment["MEMLIMIT"] = memlimit

        # - TMPDIR is the name of the only directory where the solver is allowed
        #   to read/write temporary files
        tmpdir                = getenv("TMPDIR", "/tmp")
        expanded              = expand(expanded, "TMPDIR", tmpdir)
        environment["TMPDIR"] = tmpdir

        # - DIR is the name of the directory where the solver files will be
        #   stored
        home                       = self.flags.solvers_2009_path
        expanded                   = expand(expanded, "DIR", home)
        expanded                   = expand(expanded, "HOME", home)
        environment["SOLVERSHOME"] = home

        # run the solver
        log.debug("running %s", expanded)

        (chunks, elapsed, exit_status) = run_cpu_limited(expanded, cutoff, environment)

        # analyze its output
        if exit_status is None:
            return (None, elapsed)
        else:
            for line in "".join(ct for (_, ct) in chunks).split("\n"):
                if SAT_Competition2009_Solver.__sat_line_re.match(line):
                    return (True, elapsed)
                elif SAT_Competition2009_Solver.__unsat_line_re.match(line):
                    return (False, elapsed)

            return (None, elapsed)

    @property
    def seeded(self):
        """
        Is the solver seeded?
        """

        return any("RANDOMSEED" in s for s in self.command)

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

