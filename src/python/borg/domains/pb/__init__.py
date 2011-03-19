"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import tempfile
import contextlib
import cargo
import borg

from . import opb
from . import solvers
from . import features
from . import test

logger = cargo.get_logger(__name__)

class PseudoBooleanTask(object):
    def __init__(self, path):
        self.path = path

        with borg.accounting() as accountant:
            with open(path) as opb_file:
                self.opb = opb.parse_opb_file(opb_file)

        logger.info("parsed %s in %.2f s", os.path.basename(path), accountant.total.cpu_seconds)

        self.linearized_path = None

    def get_linearized_path(self):
        if self.opb.nonlinear:
            if self.linearized_path is None:
                linearizer = os.path.join(borg.defaults.solvers_root, "PBSimple/PBlinearize")
                (linearized, _) = cargo.check_call_capturing([linearizer, self.path])
                (fd, self.linearized_path) = tempfile.mkstemp(suffix = ".opb")

                with os.fdopen(fd, "w") as linearized_file:
                    linearized_file.write(linearized)

            return self.linearized_path
        else:
            return self.path

    def clean(self):
        if self.linearized_path is not None:
            os.unlink(self.linearized_path)

            self.linearized_path = None

@borg.named_domain
class PseudoBooleanSatisfiability(object):
    name = "pb"
    extensions = ["*.opb"]

    @property
    def solvers(self):
        return solvers.named

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

