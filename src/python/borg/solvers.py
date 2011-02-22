"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re
import os.path
import itertools
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__)

def random_seed():
    """Return a random solver seed."""

    return numpy.random.randint(0, 2**31)

def prepare(command, cnf_path, budget):
    """Format command for execution."""

    keywords = {
        "root": borg.defaults.solvers_root.rstrip("/"),
        "seed": random_seed(),
        "task": cnf_path,
        "cpu_limit": budget,
        }

    return [s.format(**keywords) for s in command]

def solve_competition(command, cnf_path, budget, name = None):
    """Run a competition-compliant solver; report its cost and success."""

    # run the solver
    if name is not None:
        logger.detail("running %s for %.2f seconds", name, budget)

    prefix = [
        os.path.join(borg.defaults.solvers_root, "run"),
        "-k",
        "--time-limit={0}".format(int(round(budget))),
        ]
    command = prefix + prepare(command, cnf_path, budget)
    (stdout, stderr, code) = cargo.call_capturing(command)

    # parse the run wrapper's output
    match = re.search(r"^\[run\] time:[ \t]*(\d+.\d+) seconds$", stderr, re.M)
    (cost,) = map(float, match.groups())

    # interpret the solver's output
    if code == 10:
        answer = []

        for line in re.findall(r"^v ([ \-0-9]*)$", stdout, re.M):
            answer.extend(map(int, line.split()))

        if len(answer) <= 1 or answer[-1] != 0:
            answer = None
        else:
            answer = answer[:-1]
    elif code == 20:
        answer = False
    else:
        answer = None

    return (cost, answer)

def basic_command(relative):
    """Prepare a basic competition solver command."""

    return ["{{root}}/{0}".format(relative), "{cnf_path}", "{seed}"]

core_commands = {
    # complete
    #"precosat-570": ["{root}/precosat-570-239dbbe-100801/precosat", "--seed={seed}", "{task}"],
    "lingeling-276": ["{root}/lingeling-276-6264d55-100731/lingeling", "--seed={seed}", "{task}"],
    #"cryptominisat-2.9.0": ["{root}/cryptominisat-2.9.0Linux64", "--randomize={seed}", "{task}"],
    #"march_hi": basic_command("march_hi/march_hi"),
    # incomplete
    #"TNM": basic_command("TNM/TNM"),
    #"gnovelty+2": basic_command("gnovelty+2/gnovelty+2"),
    #"hybridGM3": basic_command("hybridGM3/hybridGM3"),
    #"adaptg2wsat++": basic_command("adaptg2wsat2009++/adaptg2wsat2009++"),
    }
satzilla_commands = {
    "SATzilla2009_R": basic_command("SATzilla2009/SATzilla2009_R"),
    }

def basic_solver(name, command):
    """Return a basic competition solver callable."""

    return cargo.curry(solve_competition, command, name = name)

named = dict(zip(core_commands, itertools.starmap(basic_solver, core_commands.items())))
satzillas = dict(zip(satzilla_commands, itertools.starmap(basic_solver, satzilla_commands.items())))

