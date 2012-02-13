"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import contextlib
import borg

from . import instance
from . import solvers
from . import features
from . import test

logger = borg.get_logger(__name__, default_level = "INFO")

class MAX_SAT_Task(object):
    def __init__(self, path):
        self.path = path

    def clean(self):
        pass

@borg.named_domain
class MAX_SAT_Domain(object):
    name = "max-sat"
    extensions = [".cnf", ".wcnf"]

    @property
    def solvers(self):
        return solvers.named

    @contextlib.contextmanager
    def task_from_path(self, task_path):
        task = MAX_SAT_Task(task_path)

        try:
            yield task
        except:
            raise
        finally:
            task.clean()

    def compute_features(self, task):
        return features.get_features_for(task.path)

    def is_final(self, task, answer):
        """Is the answer definitive for the task?"""

        return answer is not None

    def show_answer(self, task, answer):
        if answer is None:
            print "s UNKNOWN"
        else:
            (description, certificate, optimum) = answer

            if optimum is not None:
                print "o {0}".format(optimum)

            print "s {0}".format(description)

            if certificate is not None:
                print "v", " ".join(certificate)

