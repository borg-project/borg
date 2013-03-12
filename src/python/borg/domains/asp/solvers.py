"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import json
import tempfile
import contextlib
import borg

logger = borg.get_logger(__name__)

def parse_clasp_json_output(stdout):
    """Parse the output from clasp."""

    try:
        output = json.loads(stdout)
    except ValueError:
        return None

    if output["Result"] == "UNKNOWN":
        return None
    else:
        return output["Result"]

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

    def with_args(self, args):
        """Return a new solver, with extra arguments appended."""

        return ClaspSolverFactory(self._root, self._command + list(args))

def parse_clasp_human_output(stdout):
    for line in stdout.splitlines():
        if line in ["SATISFIABLE", "UNSATISFIABLE", "OPTIMUM FOUND"]:
            return line

    return None

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

def parse_yuliya_output(stdout):
    for line in stdout.splitlines():
        stripped = line.strip()

        if stripped in ["Answer: 1", "No Answer Set"]:
            return stripped

    return None

class YuliyaSolverFactory(object):
    """Construct a cmodels solver invocation."""

    def __init__(self, root, command):
        self._root = root
        self._command = command

    def __call__(self, task, stm_queue = None, solver_id = None):
        return \
            borg.solver_io.RunningSolver(
                parse_yuliya_output,
                self._command,
                self._root,
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

class LP2SAT_SolverFactory(object):
    """Construct an ASP-to-SAT solver invocation."""

    def __init__(self, root, sat_factory, domain):
        self._root = root
        self._sat_factory = sat_factory
        self._domain = domain

    def __call__(self, task, stm_queue = None, solver_id = None):
        try:
            cnf_path = task.support_paths.get("cnf-g")

            if cnf_path is None:
                (fd, cnf_path) = tempfile.mkstemp(suffix = ".cnf")

                task.support_paths["cnf-g"] = cnf_path

                with contextlib.closing(os.fdopen(fd)) as cnf_file:
                    with open(task.path, "rb") as asp_file:
                        borg.domains.asp.run_lp2sat(self._domain.binaries_path, asp_file, cnf_file)
        except borg.domains.asp.LP2SAT_FailedException:
            return borg.solver_io.EmptySolver(None)
        else:
            with borg.get_domain("sat").task_from_path(cnf_path) as sat_task:
                return self._sat_factory(sat_task)

