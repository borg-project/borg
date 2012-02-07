"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import os.path
import tempfile
import contextlib
import borg

from . import instance
from . import solvers
from . import features
from . import test

logger = borg.get_logger(__name__, default_level = "INFO")

class PseudoBooleanTask(object):
    def __init__(self, path):
        self.path = path
        self.support_paths = {}

        with open(path) as opb_file:
            self.header = instance.parse_opb_file_header(opb_file.readline())

        (self.raw_M, self.raw_N, self.nonlinear) = self.header

        if self.nonlinear:
            linearizer = os.path.join(borg.defaults.solvers_root, "PBSimple/PBlinearize")
            (linearized, _) = borg.util.check_call_capturing([linearizer, self.path])
            (fd, self.linearized_path) = tempfile.mkstemp(suffix = ".opb")
            self.support_paths["linearized"] = self.linearized_path

            with os.fdopen(fd, "w") as linearized_file:
                linearized_file.write(linearized)

            logger.info("wrote linearized instance to %s", self.linearized_path)
        else:
            self.linearized_path = path

        with borg.accounting() as accountant:
            with open(self.linearized_path) as opb_file:
                self.opb = instance.parse_opb_file_linear(opb_file)

        logger.info("parsing took %.2f s", accountant.total.cpu_seconds)

    def get_linearized_path(self):
        return self.linearized_path

    def clean(self):
        for path in self.support_paths.values():
            os.unlink(path)

        self.support_paths = {}

@borg.named_domain
class PseudoBooleanSatisfiability(object):
    name = "pb"
    extensions = [".opb", ".pbo"]

    @contextlib.contextmanager
    def task_from_path(self, task_path):
        """Clean up cached task resources on context exit."""

        task = PseudoBooleanTask(task_path)

        yield task

        task.clean()

    def compute_features(self, task):
        return features.compute_all(task.opb)

    def is_final(self, task, answer):
        """Is the answer definitive for the task?"""

        if answer is None:
            return False
        else:
            (description, _) = answer

            if task.opb.objective is None:
                return description in ("SATISFIABLE", "UNSATISFIABLE")
            else:
                return description in ("OPTIMUM FOUND", "UNSATISFIABLE")

    def show_answer(self, task, answer):
        if answer is None:
            print "s UNKNOWN"
        else:
            (description, certificate) = answer

            print "s {0}".format(description)

            if certificate is not None:
                sorted_certificate = sorted(certificate, key = lambda l: int(l[2:] if l[0] == "-" else l[1:]))

                print "v", " ".join(sorted_certificate[:task.raw_N])

