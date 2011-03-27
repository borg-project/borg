"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import re
import os.path
import cargo
import borg

logger = cargo.get_logger(__name__)

def parse_max_sat_competition(stdout):
    """Parse output from a standard competition solver."""

    optima = map(int, re.findall(r"^o +([0-9]+) *\r?$", stdout, re.M))

    if len(optima) > 0:
        optimum = optima[-1]
    else:
        optimum = None

    answer_match = re.search(r"^s +([a-zA-Z ]+) *\r?$", stdout, re.M)

    if answer_match:
        (answer_type_raw,) = answer_match.groups()
        answer_type = answer_type_raw.strip().upper()

        if answer_type == "OPTIMUM FOUND":
            certificate = []

            for line in re.findall(r"^v ([ x\-0-9]*) *\r?$", stdout, re.M):
                certificate.extend(line.split())

            if len(certificate) == 0:
                return None
        elif answer_type == "UNSATISFIABLE":
            certificate = None
        else:
            return None

        return (answer_type, certificate, optimum)

    return None

basic_commands = {
    "akmaxsat": ["{root}/akmaxsat", "{task}"],
    "sat4j-maxsat-v20101225": [
        "java",
        "-server",
        "-jar",
        "{root}/sat4j-maxsat-v20101225/sat4j-maxsat.jar",
        "{task}",
        ],
    "sat4j-maxsat-v20101225-p": [
        "java",
        "-server",
        "-jar",
        "{root}/sat4j-maxsat-v20101225/sat4j-maxsat.jar",
        "-p",
        "{task}",
        ],
    }

class MAX_SAT_BasicSolverFactory(object):
    def __init__(self, command):
        self._command = command

    def __call__(self, task, stm_queue = None, solver_id = None):
        return \
            borg.solver_io.RunningSolver(
                parse_max_sat_competition,
                self._command,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

basic_named = dict(zip(basic_commands, map(MAX_SAT_BasicSolverFactory, basic_commands.values())))
wbo_prefixes = {
    #"wbo1.4a": ["{root}/wbo1.4a", "-time-limit=1000000"],
    "wbo1.4b-fixed": ["{root}/wbo1.4b-fixed", "-time-limit=1000000"],
    }

class MAX_SAT_WBO_SolverFactory(object):
    def __init__(self, prefix):
        self._prefix = prefix

    def __call__(self, task, stm_queue = None, solver_id = None):
        (_, extension) = os.path.splitext(task.path)
        command = self._prefix + ["-file-format={0}".format(extension[1:]), "{task}"]

        return \
            borg.solver_io.RunningSolver(
                parse_max_sat_competition,
                command,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

def build_inc_solver(task, stm_queue = None, solver_id = None):
    (_, extension) = os.path.splitext(task.path)

    if extension[1:] == "cnf":
        command = ["{root}/inc-maxsatz", "{task}"]
    else:
        command = ["{root}/IncWMaxSatz", "{task}"]

    return \
        borg.solver_io.RunningSolver(
            parse_max_sat_competition,
            command,
            task.path,
            stm_queue = stm_queue,
            solver_id = solver_id,
            )

inc_named = {"inc[w]maxsatz": build_inc_solver}
wbo_named = dict(zip(wbo_prefixes, map(MAX_SAT_WBO_SolverFactory, wbo_prefixes.values())))
named = dict(basic_named.items() + wbo_named.items() + inc_named.items())

