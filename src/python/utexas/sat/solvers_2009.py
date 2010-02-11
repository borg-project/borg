"""
utexas/sat/solvers_2009.py

Run satisfiability solvers from the 2009 competition.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re

from os import getenv
from copy import copy
from os.path import (
    join,
    splitext,
    basename,
    )
from cargo.log import get_logger
from cargo.unix.accounting import run_cpu_limited
from cargo.flags import (
    Flag,
    Flags,
    )
from cargo.temporal import TimeDelta
from utexas.sat.solvers import SAT_Solver

log = get_logger(__name__)

COMMANDS_BY_NAME = {
    "adaptg2wsat2009++": [
        "HOME/main/sat09-46-li/ag++/adaptg2wsat2009++",
        "BENCHNAME",
        "RANDOMSEED",
        ],
    "CirCUs": [
        "HOME/main/sat09-42-han/CirCUs",
        "BENCHNAME",
        "-t",
        "TIMEOUT",
        ]
    "clasp": [
        "HOME/main/sat09-17-kaufmann/clasp",
        "--file=BENCHNAME",
        "--dimacs",
        "--number=1",
        "--sat-p=20,25,150",
        "--hParam=0,512",
        ]
    "glucose": [
        "HOME/main/sat09-28-audemard/glucose.sh"
        "BENCHNAME",
        ],
    "gnovelty+2":
        "HOME/main/sat09-53-pham/gnovelty+2/gnovelty+2.src",
        "BENCHNAME",
        "RANDOMSEED",
        ],
    "gNovelty+-T": [
        "HOME/multithread/sat09-57-gretton/gnovelty+-T/gnovelty+",
        "BENCHNAME",
        "RANDOMSEED",
        "1",
        ]
    "hybridGM3": [
        "HOME/main/sat09-29-balint/hybridGM3/hybridGM3",
        "BENCHNAME",
        "RANDOMSEED",
        ],
    "iPAWS": [
        "HOME/main/sat09-34-thornton/iPAWS",
        "BENCHNAME",
        "RANDOMSEED",
        ],
    "IUT_BMB_SAT": [
        "HOME/main/sat09-52-bahrami/IUT_BMB_SAT.sh",
        "BENCHNAME",
        "TMPDIR",
        ],
    "LySAT_c": [
        "HOME/main/sat09-40-sais/lysat/lysat.sh",
        "BENCHNAME",
        "c",
        ],
    "LySAT_i": [
        "HOME/main/sat09-40-sais/lysat/lysat.sh",
        "BENCHNAME",
        "i",
        ],
    "ManySAT": [
        "HOME/multithread/sat09-40-sais/manysat/manysat.sh",
        "BENCHNAME",
        "aimd",
        "1",
        ],
    "march_hi": [
        "HOME/main/sat09-19-heule/hi/march_hi",
        "BENCHNAME",
        ],
    "minisat_09z": [
        "HOME/minisat/sat09-36-iser/SatELiteGTI",
        "BENCHNAME",
        "-random-seed=RANDOMSEED",
        ],
    "minisat_cumr_p": [
        "HOME/minisat/sat09-21-masuda/cumr_p/SatELiteGTI",
        "BENCHNAME",
        ],
    "mxc_09": [
        "HOME/main/sat09-32-bregman/mxc-sat09",
        "-i",
        "BENCHNAME",
        ],
    "precosat": [
        "HOME/main/sat09-25-biere/precosat/precosat",
        "BENCHNAME",
        ],
     "rsat_09": [
        "HOME/main/sat09-13-pipatsrisawat/rsat.sh",
        "BENCHNAME",
        "TMPDIR",
        ],
     "SApperloT": [
        "HOME/main/sat09-37-kottler/base/SApperloT-base",
        "-seed=RANDOMSEED",
        "BENCHNAME",
        ],
    "SATzilla2009_C": [
        "HOME/main/sat09-30-xu/SATzilla2009_C",
        "BENCHNAME",
        ],
    "SATzilla2009_I": [
        "HOME/main/sat09-30-xu/SATzilla2009_I",
        "BENCHNAME",
        ],
    "SATzilla2009_R": [
        "HOME/main/sat09-30-xu/SATzilla2009_R",
        "BENCHNAME",
        ],
    "TNM": [
        "HOME/main/sat09-56-wei/TNM/TNM",
        "BENCHNAME",
        "RANDOMSEED",
        ],
    "VARSAT-industrial": [
        "HOME/main/sat09-24-hsu/industrial/varsat-industrial.32",
        "-seed=RANDOMSEED",
        "BENCHNAME",
        ],
    }

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

    @staticmethod
    def by_name(
        name,
        flags = class_flags.given,
        ):
        """
        Build a solver by name.
        """

        return SAT_Competition2009_Solver(COMMANDS_BY_NAME[name], flags)

