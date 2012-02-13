"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import os.path
import shutil
import tempfile
import contextlib
import borg

from . import solvers
from . import features

class GroundedAnswerSetInstance(object):
    """A grounded answer-set programming (ASP) instance."""

    def __init__(self, asp_path):
        """Initialize."""

        if asp_path.endswith(".gz"):
            with borg.util.openz(asp_path) as asp_file:
                (fd, self.path) = tempfile.mkstemp(suffix = ".asp.ground")

                final_file = os.fdopen(fd, "wb")

                shutil.copyfileobj(asp_file, final_file)

                final_file.close()

            self.unlink = True
        else:
            self.path = asp_path
            self.unlink = False

    def clean(self):
        """Clean up the grounded instance."""

        if self.unlink:
            os.unlink(self.path)

class AnswerSetProgramming(object):
    name = "asp"
    extensions = [".asp.ground", ".asp.ground.gz"]

    def __init__(self, claspre_path = None):
        if claspre_path is None:
            self._claspre_path = None
        else:
            self._claspre_path = os.path.abspath(claspre_path)

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
        if self._claspre_path is None:
            return Exception("domain not configured for static features")
        else:
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

