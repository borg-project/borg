"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import contextlib
#import borg

#solvers = borg.domains.asp.solvers
from . import solvers
from . import features

class GroundedAnswerSetInstance(object):
    """A grounded answer-set programming (ASP) instance."""

    def __init__(self, asp_path):
        """Initialize."""

        self.path = asp_path

    def clean(self):
        """Clean up the grounded instance."""

class AnswerSetProgramming(object):
    name = "asp"
    extensions = ["*.lparse"]

    def __init__(self, claspre_path):
        self._claspre_path = os.path.abspath(claspre_path)

    @contextlib.contextmanager
    def task_from_path(self, asp_path):
        """Clean up cached task resources on context exit."""

        task = GroundedAnswerSetInstance(asp_path)

        yield task

        task.clean()

    def compute_features(self, task):
        return features.get_features_for(task.path, self._claspre_path)

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

