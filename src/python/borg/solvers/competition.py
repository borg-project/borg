"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log    import get_logger
from borg.rowed   import Rowed
from borg.solvers import AbstractSolver

log = get_logger(__name__)

def scan_competition_output(lines, satisfiable = None, certificate = None):
    """
    Interpret reasonably well-formed competition-style output.
    """

    from itertools import imap

    for line in lines:
        # reported sat
        if line.startswith("s SATISFIABLE"):
            if satisfiable is not None:
                raise RuntimeError("multiple solution lines in output")
            else:
                satisfiable = True
        # reported unsat
        elif line.startswith("s UNSATISFIABLE"):
            if satisfiable is not None:
                raise RuntimeError("multiple solution lines in output")
            else:
                satisfiable = False
        # provided (part of) a sat certificate
        elif line.startswith("v "):
            literals = imap(int, line[2:].split())

            if certificate is None:
                certificate = list(literals)
            else:
                certificate.extend(literals)

    # done
    if satisfiable is None:
        return None
    else:
        from borg.sat import SAT_Answer

        return SAT_Answer(satisfiable, certificate)

class CompetitionSolver(Rowed, AbstractSolver):
    """
    A solver for that uses the circa-2009 SAT competition interface.
    """

    def __init__(
        self,
        command,
        memlimit     = None,
        solvers_home = ".",
        ):
        """
        Initialize this solver.

        @param memlimit: The memory limit reported to the solver.
        """

        # base
        Rowed.__init__(self)

        # members
        from copy import copy

        self.command      = copy(command)
        self.memlimit     = memlimit
        self.solvers_home = solvers_home

    def solve(self, task, budget, random, environment):
        """
        Execute the solver and return its outcome.
        """

        # argument sanity
        from borg.tasks import AbstractFileTask

        assert isinstance(task, AbstractFileTask)

        # set up the command arguments
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

        (without, _) = splitext(task.path)
        base         = basename(task.path)
        base_without = basename(without)
        expanded     = expand(expanded, "BENCHNAMENOPATHNOEXT", base_without)
        expanded     = expand(expanded, "BENCHNAMENOPATH", base)
        expanded     = expand(expanded, "BENCHNAMENOEXT", without)
        expanded     = expand(expanded, "BENCHNAME", task.path)

        # RANDOMSEED: a random seed which is a number between 0 and 4294967295
        from borg.solvers import get_random_seed

        if self.seeded:
            seed     = get_random_seed(random)
            expanded = expand(expanded, "RANDOMSEED", seed)
        else:
            seed = None

        # TIMELIMIT (or TIMEOUT): the total CPU time (in seconds) that the
        # solver may use before being killed
        from cargo.temporal import TimeDelta

        cutoff_s                 = TimeDelta.from_timedelta(budget).as_s
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
                    budget,
                    pty         = True,
                    environment = environment,
                    )

        # and analyze its output
        if run.exit_status is not None:
            out_lines = "".join(c for (t, c) in run.out_chunks).split("\n")
            answer    = scan_competition_output(out_lines)
        else:
            answer = None

        # return our attempt
        from borg.solvers import RunAttempt

        return RunAttempt(self, task, answer, seed, run)

    @property
    def seeded(self):
        """
        Is the solver seeded?
        """

        return any("RANDOMSEED" in s for s in self.command)

