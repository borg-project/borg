# vim: set fileencoding=UTF-8 :
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.sat.test_solvers import main

    raise SystemExit(main())

import re
import json
import numpy
import utexas.sat.solvers

from os.path            import (
    join,
    dirname,
    )
from copy               import copy
from tempfile           import NamedTemporaryFile
from cargo.io           import expandpath
from cargo.log          import get_logger
from cargo.json         import follows
from cargo.flags        import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.temporal     import TimeDelta

log = get_logger(__name__, default_level = "NOTE")

module_flags = \
    Flags(
        "Solver Execution Options",
        Flag(
            "-s",
            "--seed",
            default = "random",
            metavar = "INT",
            help    = "run solver with seed INT [%default]",
            ),
        Flag(
            "-c",
            "--cutoff",
            type    = float,
            default = 1e10,
            metavar = "FLOAT",
            help    = "run for at most FLOAT seconds [%default]",
            ),
        Flag(
            "--verify-tmpdir",
            action  = "store_true",
            help    = "verify TMPDIR compliance via strace [%default]",
            ),
        Flag(
            "-e",
            "--expected",
            metavar = "FILE",
            help    = "read solver expectations from FILE [%default]",
            ),
        Flag(
            "--tasks-dir",
            default = "",
            metavar = "PATH",
            help    = "find tasks under PATH [%default]",
            ),
        Flag(
            "-v",
            "--verbose",
            action  = "store_true",
            help    = "be noisier [%default]",
            ),
        )

def get_solver_expectations(start_path):
    """
    Retrieve descriptions of expected solver behavior.
    """

    def yield_from(raw_path):
        """
        (Recursively) yield solvers from a solvers file.
        """

        path     = expandpath(raw_path)
        relative = dirname(path)

        with open(path) as file:
            loaded = json.load(file)

        for (name, value) in loaded.get("expected", {}).items():
            yield (name, follows(value, relative))

        for included in loaded.get("includes", []):
            for pair in yield_from(expandpath(included, relative)):
                yield pair

    return dict(yield_from(start_path))

def strace_signal(lines):
    """
    Yield only strace lines that aren't obviously noise.
    """

    exclude = [
        re.compile("open\\(\".+\",.*O_RDONLY.*"),
        re.compile("--- SIG.+ ---"),
        ]

    for line in lines:
        if not any(e.search(line) for e in exclude):
            yield line

def tmpdir_uncompliance(lines):
    """
    Using output from strace, attempt to verify solver compliance.
    """

    # exclude uninteresting lines
    included = list(strace_signal(lines))

    for line in included:
        log.detail("from strace: %s", line.strip())

    # examine interesting lines
    bad  = re.compile("open\\(\"/tmp/.+\", .+\\)")
    good = re.compile("open\\(\"/tmp/solver\\_scratch\\..+\", .+\\)")

    for line in included:
        if bad.search(line) and not good.search(line):
            yield line

def do_run(solver, cnf_path, cutoff, seed_request):
    """
    Run a solver on an instance.
    """

    # if necessary, generate a seed
    if not solver.seeded:
        seed = None
    elif seed_request == "random":
        seed = numpy.random.randint(2**31 - 1)
    else:
        seed = int(seed_request)

    log.info("seed: %s", seed)

    # attempt to solve the instance
    from utexas.sat.solvers import (
        SAT_SanitizingSolver,
        SAT_UncompressingSolver,
        )

    u_solver = SAT_UncompressingSolver(SAT_SanitizingSolver(solver), solver.name)
    result   = u_solver.solve(cnf_path, cutoff, seed)

    log.info("result: %s", result)
    log.info("satisfiable? %s", result.satisfiable)
    log.info("/proc-elapsed: %s", result.run.proc_elapsed)
    log.info("usage-elapsed: %s", result.run.usage_elapsed)
    log.info("exit status: %s", result.run.exit_status)
    log.info("exit signal: %s", result.run.exit_signal)
    log.info("stdout:\n%s", "".join(c for (_, c) in result.run.out_chunks))
    log.info("stderr:\n%s", "".join(c for (_, c) in result.run.err_chunks))

    return (result.satisfiable, result.run, seed)

def do_verified_run(solver, cnf_path, cutoff, seed_request):
    """
    Run a solver, with optional verification of TMPDIR compliance.
    """

    ssolver = copy(solver)

    with NamedTemporaryFile(prefix = "strace.") as file:
        # make the solver generate a strace
        ssolver.command = ["strace", "-f", "-eopen", "-o", file.name] + ssolver.command

        # run it and read it
        result = do_run(ssolver, cnf_path, cutoff, seed_request)
        strace = file.readlines()

    # examine it
    for line in tmpdir_uncompliance(strace):
        log.warning("TMPDIR uncompliance: %s", line.strip())

    return result

def test_solver_on(solver, path, expectation):
    """
    Run a single test on a single solver.
    """

    flags = module_flags.given

    # interpret expectations
    if expectation is None:
        min_seconds  = 0.0
        max_seconds  = flags.cutoff
        seed_request = flags.seed
    else:
        # unpack the general expectation tuple
        if len(expectation) == 2:
            (should_be, time_range) = expectation
            seed_request            = flags.seed
        else:
            (should_be, time_range, seed_request) = expectation

        # unpack the expected-time tuple
        if type(time_range) is list:
            (min_seconds, max_seconds) = time_range
        else:
            min_seconds = 0.0
            max_seconds = time_range

    cutoff    = TimeDelta(seconds = max_seconds)
    full_path = join(flags.tasks_dir, expandpath(path))

    # run the solver
    if flags.verify_tmpdir:
        (outcome, run, seed) = do_verified_run(solver, full_path, cutoff, seed_request)
    else:
        (outcome, run, seed) = do_run(solver, full_path, cutoff, seed_request)

    # mark as passed/failed
    if expectation is not None:
        min_elapsed = TimeDelta(seconds = min_seconds)

        if outcome == should_be and run.proc_elapsed >= min_elapsed:
            log.note(
                u"PASS on %s: %s in %s ∈ [%s, %s+] (seed: %s)",
                path,
                outcome,
                run.proc_elapsed,
                min_elapsed,
                cutoff,
                seed,
                )
        else:
            log.warning(
                u"FAIL on %s (%s): %s in %s (seed: %s)",
                path,
                should_be,
                outcome,
                run.proc_elapsed,
                seed,
                )

def test_solver(solver, tests):
    """
    Run a set of tests on a single solver.
    """

    log.note("testing %s", solver.name)

    for (path, expectation) in tests.items():
        test_solver_on(solver, path, expectation)

def test_solvers(tests_map):
    """
    Run a set of solver tests.
    """

    for (solver, tests) in tests_map.items():
        test_solver(solver, tests)

@with_flags_parsed(
    usage = "usage: %prog [options] [solver [file]]",
    )
def main(positional):
    """
    Main.
    """

    # basic flag handling
    flags = module_flags.given

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    if flags.verbose:
        get_logger("utexas.tools.sat.run_solvers", level = "NOTSET")
        get_logger("cargo.unix.accounting", level = "DEBUG")
        get_logger("utexas.sat.solvers", level = "DEBUG")

    # build the solvers
    from utexas.sat.solvers import get_named_solvers

    solvers = get_named_solvers()

    # get expectations, if any
    if flags.expected is not None:
        expected = get_solver_expectations(flags.expected)

    # build a list of tests to run
    if len(positional) > 0:
        name = positional[0]

        if len(positional) > 1:
            tests_map = {solvers[name]: dict((p, None) for p in positional[1:])}
        else:
            tests_map = {solvers[name]: expected.get(name, {})}
    else:
        tests_map = dict((solvers[n], v) for (n, v) in expected.items())

    # and run them
    test_solvers(tests_map)

