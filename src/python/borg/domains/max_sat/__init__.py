"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import contextlib
import cargo
import borg

from . import instance
from . import solvers
#from . import features
from . import test

logger = cargo.get_logger(__name__, default_level = "INFO")

class MAX_SAT_Task(object):
    def __init__(self, path):
        self.path = path

        with borg.accounting() as accountant:
            with open(path) as task_file:
                self.instance = instance.parse_max_sat_file(task_file)

        logger.info("parsing took %.2f s", accountant.total.cpu_seconds)

@borg.named_domain
class MAX_SAT_Domain(object):
    name = "max-sat"
    extensions = ["*.cnf", "*.wcnf"]

    @property
    def solvers(self):
        return solvers.named

    @contextlib.contextmanager
    def task_from_path(self, task_path):
        yield MAX_SAT_Task(task_path)

    def compute_features(self, task):
        return features.compute_all(task.instance)

    def is_final(self, task, answer):
        """Is the answer definitive for the task?"""

        if answer is None:
            return False
        else:
            (description, _) = answer

            return description in ("OPTIMUM FOUND", "UNSATISFIABLE")

    def show_answer(self, task, answer):
        if answer is None:
            print "s UNKNOWN"
        else:
            (description, certificate) = answer

            print "s {0}".format(description)

            if certificate is not None:
                print "v", " ".join(answer)

