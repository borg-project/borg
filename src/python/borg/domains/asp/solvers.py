"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import json
import borg

logger = borg.get_logger(__name__)

# XXX are we correctly handling optimization instances?

def parse_clasp_json_output(stdout):
    """Parse the output from clasp."""

    try:
        output = json.loads(stdout)
    except ValueError:
        return None

    return output["Result"]

def parse_clasp_human_output(stdout):
    for line in stdout.splitlines():
        if line in ["SATISFIABLE", "UNSATISFIABLE", "OPTIMUM FOUND"]:
            return line

    return None

class ClaspSolverFactory(object):
    """Construct a Clasp solver invocation."""

    def __init__(self, root, command):
        self._root = root
        self._command = command

    def __call__(self, task, stm_queue = None, solver_id = None):
        return \
            borg.solver_io.RunningSolver(
                parse_clasp_json_output,
                self._command + ["--outf=2"],
                self._root,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

class ClaspfolioSolverFactory(object):
    """Construct a Claspfolio solver invocation."""

    def __init__(self, root, command, cwd):
        self._root = root
        self._command = command
        self._cwd = cwd

    def __call__(self, task, stm_queue = None, solver_id = None):
        return \
            borg.solver_io.RunningSolver(
                parse_clasp_human_output,
                self._command + ["--outf=0"],
                self._root,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                cwd = self._cwd.format(root = self._root),
                )

