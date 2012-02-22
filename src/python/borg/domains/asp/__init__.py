"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import os.path
import shutil
import resource
import tempfile
import subprocess
import contextlib
import borg

from . import solvers
from . import features

logger = borg.get_logger(__name__, default_level = "INFO")

class LP2SAT_FailedException(Exception):
    pass

def run_lp2sat(binaries_path, asp_file, cnf_file):
    """Convert a grounded ASP instance to CNF."""

    # prepare the pipeline
    commands = [
        ["lp2sat-bin/smodels-2.34", "-internal", "-nolookahead"],
        ["lp2sat-bin/lpcat-1.18"], # XXX is lpcat truly necessary?
        ["lp2sat-bin/lp2normal-1.11"],
        ["lp2sat-bin/igen-1.7"],
        ["lp2sat-bin/smodels-2.34", "-internal", "-nolookahead"],
        ["lp2sat-bin/lpcat-1.18", "-s=/dev/null"], # XXX is lpcat truly necessary?
        ["lp2sat-bin/lp2lp2-1.17", "-g"],
        ["lp2sat-bin/lp2sat-1.15", "-n"],
        ]
    full_commands = \
        [[os.path.join(binaries_path, c[0])] + c[1:] for c in commands] \
        + [["grep", "-v", "^c"]]

    # run the pipeline
    processes = []

    previous_utime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime

    try:
        with open("/dev/null", "wb") as dev_null_file:
            # start the pipeline processes
            input_pipe = asp_file

            for (i, command) in enumerate(full_commands):
                if i == len(full_commands) - 1:
                    output_pipe = cnf_file
                else:
                    output_pipe = subprocess.PIPE

                process = \
                    subprocess.Popen(
                        command,
                        stdin = input_pipe,
                        stderr = dev_null_file,
                        stdout = output_pipe,
                        )

                processes.append(process)
                input_pipe.close()

                input_pipe = process.stdout

        # wait for them to terminate
        for (process, command) in zip(processes, full_commands):
            process.wait()

            if process.returncode != 0:
                message = \
                    "process {0} in lp2sat pipeline failed: {1}".format(
                        process.pid,
                        command,
                        )

                raise LP2SAT_FailedException(message)
    except:
        logger.warning("failed to convert ASP to CNF")

        raise
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
                process.wait()

        # accumulate the cost of children
        cost = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - previous_utime

        borg.get_accountant().charge_cpu(cost)

        logger.info("lp2sat pipeline cost was %.2f s", cost)

class GroundedAnswerSetInstance(object):
    """A grounded answer-set programming (ASP) instance."""

    def __init__(self, asp_path):
        """Initialize."""

        self.support_paths = {}

        if asp_path.endswith(".gz"):
            with borg.util.openz(asp_path) as asp_file:
                (fd, self.path) = tempfile.mkstemp(suffix = ".asp.ground")

                final_file = os.fdopen(fd, "wb")

                shutil.copyfileobj(asp_file, final_file)

                final_file.close()

            self.support_paths["uncompressed"] = self.path
        else:
            self.path = asp_path

    def clean(self):
        """Clean up the grounded instance."""

        for path in self.support_paths.values():
            os.unlink(path)

        self.support_paths = {}

class AnswerSetProgramming(object):
    name = "asp"
    extensions = [".asp.ground", ".asp.ground.gz"]

    def __init__(self, binaries_path = None):
        if binaries_path is None:
            self._binaries_path = None
        else:
            self._binaries_path = os.path.abspath(binaries_path)

    @contextlib.contextmanager
    def task_from_path(self, asp_path):
        """Clean up cached task resources on context exit."""

        task = GroundedAnswerSetInstance(asp_path)

        try:
            yield task
        except:
            raise
        finally:
            task.clean()

    def compute_features(self, task):
        if self._binaries_path is None:
            return Exception("domain not configured for static features")
        else:

            return features.get_features_for(task.path, self._binaries_path)

    def is_final(self, task, answer):
        """Is the answer definitive for the task?"""

        return answer is not None

    def show_answer(self, task, answer):
        if answer is None:
            print "s UNKNOWN"

            return 0
        elif answer:
            print "s SATISFIABLE"
            print "v", " ".join(map(str, answer)), "0"

            return 10
        else:
            print "s UNSATISFIABLE"

            return 20

    @property
    def binaries_path(self):
        return self._binaries_path

