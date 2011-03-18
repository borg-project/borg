"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import borg

@borg.named_domain("pbs")
class PseudoBooleanSatisfiability(object):
    extensions = ["*.opb"]

    def __init__(self):
        self.solvers = borg.domains.solvers.pb.named

    def shorten(self, answer):
        if answer is None:
            return None
        else:
            (description, _) = answer

            return (description, None)

    def is_final(self, answer):
        return answer is not None

    def compute_features(self, task_path):
        return borg.domains.features.pb.path_compute_all(task_path)

@borg.named_domain("pbo")
class PseudoBooleanOptimization(object):
    extensions = ["*.opb"]

    def __init__(self):
        self.solvers = borg.domains.solvers.pb.named

    def shorten(self, answer):
        if answer is None:
            return None
        else:
            (description, _) = answer

            return (description, None)

    def is_final(self, answer):
        if answer is None:
            return False
        else:
            (description, _) = answer

            return description in ("OPTIMUM FOUND", "UNSATISFIABLE")

    def compute_features(self, task_path):
        return borg.domains.features.pb.path_compute_all(task_path)

