"""
utexas/sat/solvers.py

Run satisfiability solvers.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re
import json
import numpy

from os                       import (
    fsync,
    getenv,
    )
from os.path                  import (
    join,
    dirname,
    splitext,
    basename,
    )
from abc                      import abstractmethod
from copy                     import copy
from shutil                   import rmtree
from tempfile                 import (
    NamedTemporaryFile,
    mkdtemp,
    )
from itertools                import chain
from contextlib               import closing
from collections              import namedtuple
from sqlalchemy.sql.functions import (
    count,
    random as sql_random,
    )
from cargo.io                 import (
    openz,
    expandpath,
    )
from cargo.log                import get_logger
from cargo.json               import follows
from cargo.unix.accounting    import run_cpu_limited
from cargo.sugar              import ABC
from cargo.flags              import (
    Flag,
    Flags,
    )
from cargo.temporal           import TimeDelta
from utexas.sat.cnf           import write_sanitized_cnf

log          = get_logger(__name__)
module_flags = \
    Flags(
        "SAT Solver Configuration",
        Flag(
            "--solvers-file",
            default = [],
            action  = "append",
            metavar = "FILE",
            help    = "read solver descriptions from FILE [%default]",
            ),
        )

def get_random_seed(random = numpy.random):
    """
    Return a random solver seed.
    """

    return random.randint(2**31 - 1)

def get_named_solvers(paths = [], flags = {}):
    """
    Retrieve a list of named solvers.
    """

    flags = module_flags.merged(flags)

    def yield_solvers_from(raw_path):
        """
        (Recursively) yield solvers from a solvers file.
        """

        path     = expandpath(raw_path)
        relative = dirname(path)

        with open(path) as file:
            loaded = json.load(file)

        for (name, attributes) in loaded.get("solvers", {}).items():
            yield (
                name,
                SAT_CompetitionSolver(
                    attributes["command"],
                    solvers_home = relative,
                    name         = name,
                    ),
                )

        for included in loaded.get("includes", []):
            for solver in yield_solvers_from(expandpath(included, relative)):
                yield solver

    return dict(chain(*(yield_solvers_from(p) for p in chain(paths, flags.solvers_file))))

class SAT_Result(object):
    """
    Minimal outcome of a SAT solver.
    """

    def __init__(self, satisfiable):
        """
        Initialize.
        """

        self.satisfiable = satisfiable

class SAT_RunResult(SAT_Result):
    """
    Outcome of an external SAT solver binary.
    """

    def __init__(self, satisfiable, run):
        """
        Initialize.
        """

        # base
        SAT_Result.__init__(self, satisfiable)

        # members
        self.run = run

class SAT_Solver(ABC):
    """
    A solver for SAT.
    """

    def solve(self, input_path, cutoff, seed = None):
        """
        Execute the solver and return its outcome.

        @return (outcome, seconds_elapsed, seed)
        """

        with closing(openz(input_path)) as source:
            with NamedTemporaryFile(prefix = "task.", suffix = ".cnf") as temporary:
                log.info("writing %s from %s", temporary.name, input_path)

                write_sanitized_cnf(temporary, source)

                temporary.flush()

                log.info("sanitized CNF written and flushed")

                fsync(temporary.fileno())

                return self._solve(temporary.name, cutoff, seed)

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
        self.command      = copy(command)
        self.memlimit     = memlimit
        self.solvers_home = solvers_home
        self.name         = name

    def _solve(self, input_path, cutoff, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.

        Comments quote the competition specification.
        """

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
        tmpdir = None

        try:
            tmpdir                = mkdtemp(prefix = "solver.")
            expanded              = expand(expanded, "TMPDIR", tmpdir)
            environment["TMPDIR"] = tmpdir

            # run the solver
            log.debug("running %s", expanded)

            run = \
                run_cpu_limited(
                    expanded,
                    cutoff,
                    pty         = True,
                    environment = environment,
                    )
        finally:
            # clean up its mess
            if tmpdir is not None:
                rmtree(tmpdir)

        # and analyze its output
        satisfiable = None

        if run.exit_status is not None:
            for line in "".join(c for (t, c) in run.out_chunks).split("\n"):
                if SAT_CompetitionSolver.__sat_line_re.match(line):
                    satisfiable = True

                    break
                elif SAT_CompetitionSolver.__unsat_line_re.match(line):
                    satisfiable = False

                    break

        return SAT_RunResult(satisfiable, run)

    @property
    def seeded(self):
        """
        Is the solver seeded?
        """

        return any("RANDOMSEED" in s for s in self.command)

class SAT_FakeSolver(SAT_Solver):
    """
    Fake solver behavior from data.
    """

    def __init__(self):
        """
        Initialize.
        """

        pass

    def _solve(self, input_path, cutoff, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        # FIXME should instead override solve()?
        # FIXME provide a SAT_ConcreteSolver base for the _solve() solvers?

        pass

    def __get_outcome_matrix(self):
        """
        Build a matrix of outcome probabilities.
        """

        log.info("building task-action-outcome matrix")

        # hit the database
        session = ResearchSession()

        with closing(session):
            counts = numpy.zeros((self.ntasks, self.nactions, self.noutcomes))

            for action in self.actions:
                run_case  = case([(SAT_SolverRun.proc_elapsed <= action.cutoff, SAT_SolverRun.satisfiable)])
                statement =                                                           \
                    select(
                        [
                            SAT_SolverRun.task_uuid,
                            run_case.label("result"),
                            count(),
                            ],
                        and_(
                            SAT_SolverRun.task_uuid.in_([t.task.uuid for t in self.tasks]),
                            SAT_SolverRun.solver        == action.solver,
                            SAT_SolverRun.cutoff        >= action.cutoff,
                            ),
                        )                                                             \
                        .group_by(SAT_SolverRun.task_uuid, "result")
                executed  = session.connection().execute(statement)

                # build the matrix
                world_tasks = dict((t.task.uuid, t) for t in self.tasks)
                total_count = 0

                for (task_uuid, result, nrows) in executed:
                    # map storage instances to world instances
                    world_task    = world_tasks[task_uuid]
                    world_outcome = SAT_Outcome.BY_VALUE[result]

                    # record the outcome count
                    counts[world_task.n, action.n, world_outcome.n]  = nrows
                    total_count                                     += nrows

                if total_count == 0:
                    log.warning("no rows found for action %s", action, t.task.uuid)

            norms = numpy.sum(counts, 2, dtype = numpy.float)

            return counts / norms[:, :, numpy.newaxis]

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

