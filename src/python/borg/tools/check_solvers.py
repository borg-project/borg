# vim: set fileencoding=UTF-8 :
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                     import call
    from borg.tools.check_solvers import main

    call(main)

from plac      import annotations
from cargo.log import get_logger

log = get_logger(__name__, default_level = "NOTE")

def get_solver_expectations(start_path):
    """
    Retrieve descriptions of expected solver behavior.
    """

    def yield_from(raw_path):
        """
        (Recursively) yield solvers from a solvers file.
        """

        from os.path    import dirname
        from cargo.io   import expandpath
        from cargo.json import (
            follows,
            load_json,
            )

        path     = expandpath(raw_path)
        relative = dirname(path)
        loaded   = load_json(path)

        for (name, value) in loaded.get("expected", {}).items():

        for included in loaded.get("includes", []):
            for pair in yield_from(expandpath(included, relative)):
                yield pair

    return dict(yield_from(start_path))

def strace_signal(lines):
    """
    Yield only strace lines that aren't obviously noise.
    """

    from re import compile

    exclude = [
        compile("open\\(\".+\",.*O_RDONLY.*"),
        compile("--- SIG.+ ---"),
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
    from re import compile

    bad  = compile("open\\(\"/tmp/.+\", .+\\)")
    good = compile("open\\(\"/tmp/solver\\_scratch\\..+\", .+\\)")

    for line in included:
        if bad.search(line) and not good.search(line):
            yield line

def do_run(solver, cnf_path, cutoff, seed_request):
    """
    Run a solver on an instance.
    """

    # if necessary, generate a seed
    import numpy

    if not solver.seeded:
        seed = None
    elif seed_request == "random":
        seed = numpy.random.randint(2**31 - 1)
    else:
        seed = int(seed_request)

    log.info("seed: %s", seed)

    # attempt to solve the instance
    from os.path            import join
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

    from copy     import copy
    from tempfile import NamedTemporaryFile

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

def check_solver_on(solver, path, expectation, cutoff, seed_flag, tasks_dir, verify_tmpdir):
    """
    Run a single test on a single solver.
    """

    from os.path  import join
    from datetime import timedelta
    from cargo.io import expandpath

    # interpret expectations
    if expectation is None:
        min_seconds  = 0.0
        max_seconds  = cutoff
        seed_request = seed_flag
    else:
        # unpack the general expectation tuple
        if len(expectation) == 2:
            (should_be, time_range) = expectation
            seed_request            = seed_flag
        else:
            (should_be, time_range, seed_request) = expectation

        # unpack the expected-time tuple
        if type(time_range) is list:
            (min_seconds, max_seconds) = time_range
        else:
            min_seconds = 0.0
            max_seconds = time_range

    budget    = timedelta(seconds = max_seconds)
    full_path = join(tasks_dir, expandpath(path))

    # run the solver
    if verify_tmpdir:
        (outcome, run, seed) = do_verified_run(solver, full_path, budget, seed_request)
    else:
        (outcome, run, seed) = do_run(solver, full_path, budget, seed_request)

    # mark as passed/failed
    if expectation is not None:
        min_elapsed = timedelta(seconds = min_seconds)

        if outcome == should_be and run.proc_elapsed >= min_elapsed:
            log.note(
                u"PASS on %s: %s in %s ∈ [%s, %s+] (seed: %s)",
                path,
                outcome,
                run.proc_elapsed,
                min_elapsed,
                budget,
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

@annotations(
    solver        = ("solver to check"          , "positional"),
    seed          = ("solver PRNG seed"         , "option"     , "s" , int)  ,
    cutoff        = ("max run duration"         , "option"     , "c" , float),
    verify_tmpdir = ("verify $TMPDIR compliance", "flag")      ,
    expected      = ("read expectations from"   , "option"     , "e"),
    tasks_dir     = ("find tasks under"         , "option")    ,
    verbose       = ("be noisier"               , "flag"       , "v"),
    )
def main(
    solver        = None,
    seed          = "random",
    cutoff        = 1e10,
    verify_tmpdir = False,
    expected      = None,
    tasks_dir     = "",
    verbose       = False,
    *files,
    ):
    """
    Main.
    """

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    if verbose:
        get_logger("cargo.unix.accounting", level = "DEBUG")

    # build the solvers
    from utexas.sat.solvers import get_named_solvers

    solvers = get_named_solvers()

    # get expectations, if any
    if expected is not None:
        expected = get_solver_expectations(expected)

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
    for (solver, tests) in tests_map.items():
        log.note("testing %s", solver.name)

        for (path, expectation) in tests.items():
            check_solver_on(solver, path, expectation, cutoff, seed, tasks_dir, verify_tmpdir)

