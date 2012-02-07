"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import re
import os.path
import borg

logger = borg.get_logger(__name__)

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

class MAX_SAT_BasicSolverFactory(object):
    def __init__(self, root, command):
        self._root = root
        self._command = command

    def __call__(self, task, stm_queue = None, solver_id = None):
        return \
            borg.solver_io.RunningSolver(
                parse_max_sat_competition,
                self._command,
                self._root,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

class MAX_SAT_WBO_SolverFactory(object):
    def __init__(self, root, prefix):
        self._root = root
        self._prefix = prefix

    def __call__(self, task, stm_queue = None, solver_id = None):
        (_, extension) = os.path.splitext(task.path)
        command = self._prefix + ["-file-format={0}".format(extension[1:]), "{task}"]

        return \
            borg.solver_io.RunningSolver(
                parse_max_sat_competition,
                command,
                self._root,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

class MAX_SAT_IncSatzSolverFactory(object):
    def __init__(self, root, (inc_command, incw_command)):
        self._root = root
        self._inc_command = inc_command
        self._incw_command = incw_command

    def __call__(self, task, stm_queue = None, solver_id = None):
        (_, extension) = os.path.splitext(task.path)

        if extension[1:] == "cnf":
            command = self._inc_command
        else:
            command = self._incw_command

        return \
            borg.solver_io.RunningSolver(
                parse_max_sat_competition,
                command,
                self._root,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

