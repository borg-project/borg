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
        os.path.join(borg.defaults.solvers_root, "run-1.4/run"),
        "-k",
        "--time-limit={0}".format(int(round(budget))),
        ]
    command = prefix + prepare(command, cnf_path, budget)
    (stdout, stderr, code) = cargo.call_capturing(command)

    # parse the run wrapper's output
    match = re.search(r"^\[run\] time:[ \t]*(\d+.\d+) seconds$", stderr, re.M)
    (cost,) = map(float, match.groups())

    # parse the solver's output
    match = re.search(r"^s +(.+)$", stdout, re.M)

    if match:
        (answer_type,) = map(str.upper, match.groups())

        if answer_type == "SATISFIABLE":
            answer = []

            for line in re.findall(r"^v ([ \-0-9]*)$", stdout, re.M):
                answer.extend(map(int, line.split()))

            with open(cnf_path) as cnf_file:
                (_, N, _) = borg.dimacs.parse_cnf_header(cnf_file)

            if len(answer) == N + 1 and answer[-1] == 0:
                answer = answer[:-1]
            elif len(answer) != N:
                answer = None

            if answer is None:
                logger.warning("missing or invalid certificate from subsolver")
        elif answer_type == "UNSATISFIABLE":
            answer = False
        else:
            answer = None
    else:
        answer = None

    return (cost, answer)

def basic_command(relative):
    """Prepare a basic competition solver command."""

    return ["{{root}}/{0}".format(relative), "{task}", "{seed}"]

core_commands = {
    # complete
    #"precosat-570": ["{root}/precosat-570-239dbbe-100801/precosat", "--seed={seed}", "{task}"],
    #"lingeling-276": ["{root}/lingeling-276-6264d55-100731/lingeling", "--seed={seed}", "{task}"],
    #"cryptominisat-2.9.0": ["{root}/cryptominisat-2.9.0/cryptominisat-2.9.0Linux64", "--randomize={seed}", "{task}"],
    #"glucosER": ["{root}/glucosER/glucoser_static", "{task}"],
    #"glucose": ["{root}/glucose/glucose_static", "{task}"],
    #"SApperloT": ["{root}/SApperloT/SApperloT-base", "-seed={seed}", "{task}"],
    "march_hi": basic_command("march_hi/march_hi"),
    "kcnfs-2006": ["{root}/kcnfs-2006/kcnfs-2006", "{task}"],
    # incomplete
    "TNM": basic_command("TNM/TNM"),
    "gnovelty+2": basic_command("gnovelty+2/gnovelty+2"),
    "hybridGM3": basic_command("hybridGM3/hybridGM3"),
    "adaptg2wsat++": basic_command("adaptg2wsat2009++/adaptg2wsat2009++"),
    "iPAWS": basic_command("iPAWS/iPAWS"),
    "FH": basic_command("FH/FH"),
    "NCVWr": basic_command("NCVWr/NCVWr"),
    }
satzilla_commands = {
    #"SATzilla2009_R": basic_command("SATzilla2009/SATzilla2009_R"),
    "SATzilla2009_I": basic_command("SATzilla2009/SATzilla2009_I"),
    }

def basic_solver(name, command):
    """Return a basic competition solver callable."""

    return cargo.curry(solve_competition, command, name = name)

named = dict(zip(core_commands, itertools.starmap(basic_solver, core_commands.items())))
satzillas = dict(zip(satzilla_commands, itertools.starmap(basic_solver, satzilla_commands.items())))

