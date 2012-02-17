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
    """A pseudo-Boolean satisfiability (PB) instance."""

    def __init__(self, path, linearizer_path = None):
        self.path = path
        self.support_paths = {}

        with open(path) as opb_file:
            self.header = instance.parse_opb_file_header(opb_file.readline())

        (self.raw_M, self.raw_N, self.nonlinear) = self.header

        if self.nonlinear:
            assert linearizer_path is not None

            (linearized, _) = borg.util.check_call_capturing([linearizer_path, self.path])
            (fd, self.linearized_path) = tempfile.mkstemp(suffix = ".opb")
            self._was_linearized = True

            self.support_paths["linearized"] = self.linearized_path

            with os.fdopen(fd, "w") as linearized_file:
                linearized_file.write(linearized)

            logger.info("wrote linearized instance to %s", self.linearized_path)
        else:
            self.linearized_path = path
            self._was_linearized = False

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

    def __init__(self, linearizer_path = None):
        if linearizer_path is None:
            self._linearizer_path = None
        else:
            self._linearizer_path = os.path.abspath(linearizer_path)

    @contextlib.contextmanager
    def task_from_path(self, task_path):
        """Clean up cached task resources on context exit."""

        task = PseudoBooleanTask(task_path, linearizer_path = self._linearizer_path)

        try:
            yield task
        except:
            raise
        finally:
            task.clean()

    def compute_features(self, task):
        """Compute static features of the given task."""

        (names, values) = features.compute_all(task.opb)

        names.append("nonlinear")
        values.append(1.0 if task._was_linearized else -1.0)

        return (names, values)

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

