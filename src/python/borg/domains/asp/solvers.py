"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import json
import cargo
import borg

logger = cargo.get_logger(__name__)

def parse_asp_output(stdout):
    """Parse the output from clasp."""

    try:
        output = json.loads(stdout)
    except ValueError:
        return None

    return output["Result"]

class ClaspSolverFactory(object):
    """Construct a Clasp solver invocation."""

    def __init__(self, root, command):
        """Initialize."""

        self._root = root
        self._command = command

    def __call__(self, task, stm_queue = None, solver_id = None):
        return \
            borg.solver_io.RunningSolver(
                parse_asp_output,
                self._command + ["--outf=2"],
                self._root,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

