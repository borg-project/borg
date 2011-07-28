"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import cargo

logger = cargo.get_logger(__name__, default_level = "INFO")

class RunRecord(object):
    """Record of a solver run."""

    def __init__(self, solver, budget, cost, success):
        """Initialize."""

        self.solver = solver
        self.budget = budget
        self.cost = cost
        self.success = success

class TrainingData(object):
    """Load and access portfolio training data."""

    def __init__(self, task_paths, domain, suffix):
        """Collect training data from task paths."""

        self._run_lists = {}
        self._feature_vectors = {}

        for path in task_paths:
            # load run records
            run_list = []
            run_data = numpy.recfromcsv(path + suffix, usemask = True)
            rows = run_data.tolist()

            if run_data.shape == ():
                rows = [rows]

            for (run_solver, run_budget, run_cost, run_succeeded, run_answer) in rows:
                record = RunRecord(run_solver, run_budget, run_cost, run_succeeded)

                run_list.append(record)

            self._run_lists[path] = run_list

            # load feature data
            feature_vector = numpy.recfromcsv("{0}.features.csv".format(path)).tolist()

            self._feature_vectors[path] = feature_vector

    def get_run_list(self, id_):
        """Retrieve runs made on a task."""

        return self._run_lists[id_]

    def get_run_lists(self):
        """Retrieve runs made on all tasks."""

        return self._run_lists

    def get_feature_vector(self, id_):
        """Retrieve features of a task."""

        return self._feature_vectors[id_]

    def get_feature_vectors(self):
        """Retrieve features of all tasks."""

        return self._feature_vectors

    @staticmethod
    def from_roots(tasks_roots, domain, suffix = ".runs.csv"):
        """Collect training data by scanning for tasks."""

        task_paths = []

        for tasks_root in tasks_roots:
            task_paths.extend(cargo.files_under(tasks_root, domain.extensions))

        return TrainingData(task_paths)

def outcome_matrices_from_runs(solver_index, budgets, run_lists):
    """Build run-outcome matrices from records."""

    S = len(solver_index)
    B = len(budgets)
    N = len(run_lists)
    successes = numpy.zeros((N, S, B))
    attempts = numpy.zeros((N, S))

    for (n, (_, runs)) in enumerate(run_lists.items()):
        for run in runs:
            s = solver_index.get(run.solver)

            if s is not None and run.budget >= budgets[-1]:
                b = numpy.digitize([run.cost], budgets)

                attempts[n, s] += 1.0

                if b < B and run.success:
                    successes[n, s, b] += 1.0

    return (successes, attempts)

