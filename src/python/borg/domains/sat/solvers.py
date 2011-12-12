"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import re
import borg

logger = borg.get_logger(__name__)

def parse_sat_output(stdout):
    """Parse a solver's standard competition-format output."""

    match = re.search(r"^s +(.+)$", stdout, re.M)

    if match:
        (answer_type,) = map(str.upper, match.groups())

        if answer_type == "SATISFIABLE":
            answer = []

            for line in re.findall(r"^v ([ \-0-9]*)$", stdout, re.M):
                answer.extend(map(int, line.split()))

            if answer[-1] == 0:
                return answer[:-1]
        elif answer_type == "UNSATISFIABLE":
            return False

    return None

class SAT_SolverFactory(object):
    """Construct a basic competition solver callable."""

    def __init__(self, root, command, library_paths = ()):
        """Initialize."""

        self._root = root
        self._command = command

    def __call__(self, task, stm_queue = None, solver_id = None):
        return \
            borg.solver_io.RunningSolver(
                parse_sat_output,
                self._command,
                self._root,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

