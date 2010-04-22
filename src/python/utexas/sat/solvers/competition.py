"""
utexas/sat/solvers/competition.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.solvers.base import SAT_Solver

class SAT_RunResult(SAT_Result):
    """
    Outcome of an external SAT solver binary.
    """

    def __init__(self, satisfiable, certificate, run):
        """
        Initialize.
        """

        SAT_Result.__init__(self)

        self._satisfiable = satisfiable
        self._certificate = certificate
        self.run         = run

    @abstractproperty
    def satisfiable(self):
        """
        Did the solver report the instance satisfiable?
        """

        return self._satisfiable

    @abstractproperty
    def certificate(self):
        """
        Certificate of satisfiability, if any.
        """

        return self._certificate

class SAT_CompetitionSolver(SAT_Solver):
    """
    A solver for SAT that uses the circa-2009 competition interface.
    """

    def __init__(
        self,
        command,
        memlimit     = None,
        solvers_home = ".",
        name         = None,
        ):
        """
        Initialize this solver.

        @param memlimit: The memory limit reported to the solver.
        """

        # base
        SAT_Solver.__init__(self)

        # members
        from copy import copy

        self.command      = copy(command)
        self.memlimit     = memlimit
        self.solvers_home = solvers_home
        self.name         = name

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.

        Comments quote the competition specification.
        """

        # FIXME support no-cutoff operation

        def expand(strings, variable, value):
            """
            Expand occurences of variable in string with value.
            """

            return [s.replace(variable, str(value)) for s in strings]

        # expand variables in an element of a solver command string. note that
        # the order of variable expansion matters: longer variables must be
        # expanded first, to avoid expanding matching shorter substrings.
        expanded    = self.command
        environment = {}

        # BENCHNAME: the name of the file (with both path and extension)
        #            containing the instance to solve
        # BENCHNAMENOEXT: name of the file with path but without extension),
        # BENCHNAMENOPATH: name of the file without path but with extension
        # BENCHNAMENOPATHNOEXT: name of the file without path nor extension
        from os.path import (
            splitext,
            basename,
            )

        (without, _) = splitext(input_path)
        base         = basename(input_path)
        base_without = basename(without)
        expanded     = expand(expanded, "BENCHNAMENOPATHNOEXT", base_without)
        expanded     = expand(expanded, "BENCHNAMENOPATH", base)
        expanded     = expand(expanded, "BENCHNAMENOEXT", without)
        expanded     = expand(expanded, "BENCHNAME", input_path)

        # RANDOMSEED: a random seed which is a number between 0 and 4294967295
        if seed is not None:
            expanded = expand(expanded, "RANDOMSEED", seed)
        elif any("RANDOMSEED" in s for s in self.command):
            raise ValueError("no seed provided for seeded solver")

        # TIMELIMIT (or TIMEOUT): the total CPU time (in seconds) that the
        # solver may use before being killed
        from cargo.temporal import TimeDelta

        cutoff_s                 = TimeDelta.from_timedelta(cutoff).as_s
        expanded                 = expand(expanded, "TIMELIMIT", cutoff_s)
        expanded                 = expand(expanded, "TIMEOUT", cutoff_s)
        environment["TIMELIMIT"] = "%.1f" % cutoff_s
        environment["TIMEOUT"]   = environment["TIMELIMIT"]

        # SATTIMEOUT is a synonym for TIMELIMIT used by pre-2009 solvers
        environment["SATTIMEOUT"] = environment["TIMELIMIT"]

        # only report memlimit if requested to do so
        if self.memlimit is not None:
            # MEMLIMIT: the total amount of memory (in MiB) that the solver may use
            memlimit                = 2048
            expanded                = expand(expanded, "MEMLIMIT", memlimit)
            environment["MEMLIMIT"] = memlimit

            # SATRAM is a synonym for MEMLIMIT potentially used by pre-2009 solvers
            environment["SATRAM"]   = environment["MEMLIMIT"]

        # HOME: where the solver files are stored
        expanded                   = expand(expanded, "HERE", self.solvers_home)
        environment["SOLVERSHOME"] = self.solvers_home

        # TMPDIR: the only directory where the solver is allowed to read/write
        # temporary files
        from cargo.io import mkdtemp_scoped

        with mkdtemp_scoped(prefix = "solver_scratch.") as tmpdir:
            expanded              = expand(expanded, "TMPDIR", tmpdir)
            environment["TMPDIR"] = tmpdir

            # run the solver
            from cargo.unix.accounting import run_cpu_limited

            log.note("running %s", expanded)

            run = \
                run_cpu_limited(
                    expanded,
                    cutoff,
                    pty         = True,
                    environment = environment,
                    )

        # and analyze its output
        satisfiable = None
        certificate = None

        if run.exit_status is not None:
            from itertools import imap

            for line in "".join(c for (t, c) in run.out_chunks).split("\n"):
                # reported sat
                if line.startswith("s SATISFIABLE"):
                    if satisfiable is not None:
                        raise SolverError("multiple solution lines in solver output")
                    else:
                        satisfiable = True
                # reported unsat
                elif line.startswith("s UNSATISFIABLE"):
                    if satisfiable is not None:
                        raise SolverError("multiple solution lines in solver output")
                    else:
                        satisfiable = False
                # provided (part of) a sat certificate
                elif line.startswith("v "):
                    literals = imap(int, line[2:].split())

                    if certificate is None:
                        certificate = list(literals)
                    else:
                        certificate.extend(literals)

        # crazy solver?
        if satisfiable is True:
            if certificate is None:
                raise SolverError("solver reported sat but did not provide certificate")
        elif certificate is not None:
            raise SolverError("solver did not report sat but provided certificate")

        return SAT_RunResult(satisfiable, certificate, run)

    @property
    def seeded(self):
        """
        Is the solver seeded?
        """

        return any("RANDOMSEED" in s for s in self.command)

