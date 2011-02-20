"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.run_solvers import main

    plac.call(main)

import re
import os.path
import csv
import numpy
import cargo

logger = cargo.get_logger(__name__, default_level = "INFO")

def solve_competition(command, cnf_path, budget):
    """Run the CryptoMiniSat solver; report its cost and success."""

    command_prefix = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/run-1.4/run",
        "-k",
        "--time-limit={0}".format(int(round(budget))),
        ]

    (stdout, stderr, code) = cargo.call_capturing(command_prefix + command)

    match = re.search(r"^\[run\] time:[ \t]*(\d+.\d+) seconds$", stderr, re.M)
    (cost,) = map(float, match.groups())

    if code == 10:
        answer = True
    elif code == 20:
        answer = False
    else:
        answer = None

    print stdout
    print stderr

    return (cost, answer)

def solve_tnm(cnf_path, budget):
    """Run the TNM solver; report its cost and success."""

    seed = numpy.random.randint(0, 2**31)
    command = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/TNM/TNM",
        cnf_path,
        str(seed),
        ]

    return (seed,) + solve_competition(command, cnf_path, budget)

def solve_cryptominisat(cnf_path, budget):
    """Run the CryptoMiniSat solver; report its cost and success."""

    seed = numpy.random.randint(0, 2**31)
    command = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/cryptominisat-2.9.0Linux64",
        "--randomize={0}".format(seed),
        cnf_path,
        ]

    return (seed,) + solve_competition(command, cnf_path, budget)

def solve_march_hi(cnf_path, budget):
    """Run the solver; report its cost and success."""

    seed = numpy.random.randint(0, 2**31)
    command = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/march_hi",
        cnf_path,
        str(seed),
        ]

    return (seed,) + solve_competition(command, cnf_path, budget)

def solve_satzilla2009_r(cnf_path, budget):
    """Run the solver; report its cost and success."""

    seed = numpy.random.randint(0, 2**31)
    command = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/SATzilla2009/SATzilla2009_R",
        cnf_path,
        str(seed),
        ]

    return (seed,) + solve_competition(command, cnf_path, budget)

def solve_gnoveltyplus2(cnf_path, budget):
    """Run the solver; report its cost and success."""

    seed = numpy.random.randint(0, 2**31)
    command = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/gnovelty+2/gnovelty+2",
        cnf_path,
        str(seed),
        ]

    return (seed,) + solve_competition(command, cnf_path, budget)

def solve_hybridgm3(cnf_path, budget):
    """Run the solver; report its cost and success."""

    seed = numpy.random.randint(0, 2**31)
    command = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/hybridGM3/hybridGM3",
        cnf_path,
        str(seed),
        ]

    return (seed,) + solve_competition(command, cnf_path, budget)

def solve_adaptg2wsat2009plusplus(cnf_path, budget):
    """Run the solver; report its cost and success."""

    seed = numpy.random.randint(0, 2**31)
    command = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/adaptg2wsat2009++/adaptg2wsat2009++",
        cnf_path,
        str(seed),
        ]

    return (seed,) + solve_competition(command, cnf_path, budget)

solve_methods = {
    #"TNM": solve_tnm,
    #"cryptominisat-2.9.0": solve_cryptominisat,
    #"march_hi": solve_march_hi,
    #"SATzilla2009_R": solve_satzilla2009_r,
    #"gnovelty+2": solve_gnoveltyplus2,
    #"hybridGM3": solve_hybridgm3,
    "adaptg2wsat++": solve_adaptg2wsat2009plusplus,
    }

def run_solver_on(solver_name, cnf_path, budget):
    """Run a solver."""

    solve = solve_methods[solver_name]
    (seed, cost, answer) = solve(cnf_path, budget)

    logger.info("%s reported %s in %.2f (of %.2f) on %s", solver_name, answer, cost, budget, cnf_path)

    return (solver_name, seed, budget, cost, answer)

@plac.annotations(
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(tasks_root, workers = 0):
    """Collect solver running-time data."""

    cargo.enable_default_logging()

    def yield_runs():
        paths = list(cargo.files_under(tasks_root, ["*.cnf"]))

        for _ in xrange(4):
            for solver_name in solve_methods:
                for path in paths:
                    yield (run_solver_on, [solver_name, path, 6000.0])

    def collect_run((_, arguments), row):
        (_, cnf_path, _) = arguments
        csv_path = cnf_path + ".rtd.csv"
        existed = os.path.exists(csv_path)

        with open(csv_path, "a") as csv_file:
            writer = csv.writer(csv_file)

            if not existed:
                writer.writerow(["solver", "seed", "budget", "cost", "answer"])

            #writer.writerow(row)

    cargo.distribute_or_labor(yield_runs(), workers, collect_run)

