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
from tempfile                 import mkdtemp
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
    decompress_if,
    guess_encoding,
    mkdtemp_scoped,
    )
from cargo.log                import get_logger
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

def get_random_seed(random):
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

class SolverError(RuntimeError):
    """
    The solver failed in an unexpected way.
    """

class SAT_Result(object):
    """
    Minimal outcome of a SAT solver.
    """

    @abstractproperty
    def satisfiable(self):
        """
        Did the solver report the instance satisfiable?
        """

    @abstractproperty
    def certificate(self):
        """
        Certificate of satisfiability, if any.
        """

class SAT_Solver(ABC):
    """
    A solver for SAT.
    """

    @abstractmethod
    def solve(self, input_path, cutoff = None, seed = None):
        """
        Attempt to solve the specified instance; return the outcome.
        """

class SAT_UncompressingSolver(SAT_Solver):
    """
    Execute another solver using an uncompressed instance.
    """

    def __init__(self, solver):
        """
        Initialize.
        """

        SAT_Solver.__init__(self)

        self.solver = solver

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Attempt to solve the specified instance; return the outcome.
        """

        # FIXME only create the temporary directory if necessary

        with mkdtemp_scoped(prefix = "solver_input.") as sandbox_path:
            # decompress the instance, if necessary
            uncompressed_path = \
                decompress_if(
                    input_path,
                    join(sandbox_path, "uncompressed.cnf"),
                    )

            log.info("uncompressed task is %s", uncompressed_path)

            # execute the next solver in the chain
            return self.solver.solve(uncompressed_path, cutoff, seed)

class SAT_SanitizingSolver(SAT_Solver):
    """
    Execute another solver using a sanitized instance.
    """

    def __init__(self, solver):
        """
        Initialize.
        """

        SAT_Solver.__init__(self)

        self.solver = solver

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Attempt to solve the specified instance; return the outcome.
        """

        log.info("starting to solve %s", input_path)

        # FIXME use a temporary file, not directory

        with mkdtemp_scoped(prefix = "solver_input.") as sandbox_path:
            # unconditionally sanitize the instance
            sanitized_path = join(sandbox_path, "sanitized.cnf")

            with open(uncompressed_path) as uncompressed_file:
                with open(sanitized_path, "w") as sanitized_file:
                    write_sanitized_cnf(uncompressed_file, sanitized_file)
                    sanitized_file.flush()
                    fsync(sanitized_file.fileno())

            log.info("sanitized task is %s", sanitized_path)

            # execute the next solver in the chain
            return self.solver.solve(sanitized_path, cutoff, seed)

class SAT_PreprocessingSolverResult(SAT_Result):
    """
    Outcome of a solver with a preprocessing step.
    """

    def __init__(self, preprocessor_output, solver_result, certificate):
        """
        Initialize.
        """

        SAT_Result.__init__(self)

        self.preprocessor_output = preprocessor_output
        self.solver_result       = solver_result
        self.satisfiable         = solver_result.satisfiable
        self.certificate         = certificate

class SAT_PreprocessingSolver(SAT_Solver):
    """
    Execute a solver after a preprocessor pass.
    """

    def __init__(self, preprocessor, solver):
        """
        Initialize.
        """

        SAT_Solver.__init__(self)

        self.preprocessor = preprocessor
        self.solver       = solver

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        with mkdtemp_scoped(prefix = "sat_preprocessing.") as sandbox_path:
            preprocessed = self.preprocessor.preprocess(input_path, sandbox_path, cutoff)
            remaining    = max(TimeDelta(), cutoff - preprocess.elapsed)
            result       = self.solver.solve(preprocessed.cnf_path, remaining, seed)

            if result.certificate is None:
                extended = None
            else:
                extended = preprocessed.extend(result.certificate)

            return SAT_PreprocessingSolverResult(preprocessed, result, extended)

class SAT_RunResult(SAT_Result):
    """
    Outcome of an external SAT solver binary.
    """

    def __init__(self, satisfiable, certificate, run):
        """
        Initialize.
        """

        SAT_Result.__init__(self)

        self.satisfiable = satisfiable
        self.certificate = certificate
        self.run         = run

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
        with mkdtemp_scoped(prefix = "solver_scratch.") as tmpdir:
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

class SAT_FakeSolver(SAT_Solver):
    """
    Fake solver behavior from data.
    """

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        # FIXME provide a SAT_ConcreteSolver base for the _solve() solvers?
        # FIXME support no-cutoff operation

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

    def solve(self, input_path, cutoff, seed = None):
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

