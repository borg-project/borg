"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re
import os.path
import numpy
import cargo

logger = cargo.get_logger(__name__, default_level = "INFO")

def random_seed():
    """Return a random solver seed."""

    return numpy.random.randint(0, 2**31)

def solve_competition(prepare, cnf_path, budget):
    """Run a standard competition solver; report its cost and success."""

    # run the solver
    prefix = [
        "/scratch/cluster/bsilvert/sat-competition-2011/solvers/run-1.4/run",
        os.path.join(borg.defaults.solvers_root, "run"),
        "-k",
        "--time-limit={0}".format(int(round(budget))),
        ]
    command = prefix + prepare(cnf_path, budget)
    (stdout, stderr, code) = cargo.call_capturing(command)

    # parse its output
    match = re.search(r"^\[run\] time:[ \t]*(\d+.\d+) seconds$", stderr, re.M)
    (cost,) = map(float, match.groups())

    if code == 10:
        answer = True
    elif code == 20:
        answer = False
    else:
        answer = None

    return (cost, answer)

def solve_cryptominisat(cnf_path, budget):
    """Run the CryptoMiniSat solver; report its cost and success."""

    command = [
        os.path.join(borg.defaults.solvers_root, "cryptominisat-2.9.0Linux64"),
        "--randomize={0}".format(random_seed()),
        cnf_path,
        ]

    return solve_competition(command, cnf_path, budget)

def prepare_basic(relative, cnf_path, budget):
    """Prepare a basic competition solver command."""

    return [
        os.path.join(borg.defaults.solvers_root, relative),
        cnf_path,
        str(random_seed()),
        ]

def basic_solver(relative):
    """Return a basic competition solver callable."""

    return cargo.curry(solve_competition, cargo.curry(prepare_basic, relative))

named = {
    "TNM": basic_solver("TNM/TNM"),
    "cryptominisat-2.9.0": solve_cryptominisat,
    "march_hi": basic_solver("march_hi/march_hi"),
    "SATzilla2009_R": basic_solver("SATzilla2009/SATzilla2009_R"),
    "gnovelty+2": basic_solver("gnovelty+2/gnovelty+2"),
    "hybridGM3": basic_solver("hybridGM3/hybridGM3"),
    "adaptg2wsat++": basic_solver("adaptg2wsat2009++/adaptg2wsat2009++"),
    }

