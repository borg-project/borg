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

    def __init__(self):
        """Initialize."""

        self.run_lists = {}
        self.feature_vectors = {}

    def add_run(self, id_, run):
        """Add a run to these data."""

        runs = self.run_lists.get(id_)

        if runs is None:
            self.run_lists[id_] = [run]
        else:
            runs.append(run)

    def get_run_count(self):
        """Return the number of runs stored."""

        return sum(map(len, self.run_lists.values()))

    def add_feature_vector(self, id_, vector):
        """Add a feature vector to these data."""

        assert id_ not in self.feature_vectors

        self.feature_vectors[id_] = vector

    def get_feature_vector(self, id_):
        """Retrieve features of a task."""

        return self.feature_vectors[id_]

    def get_feature_vectors(self):
        """Retrieve features of all tasks."""

        return self.feature_vectors

    def get_common_budget(self):
        """Retrieve the common run budget, if any."""

        budget = None

        for runs in self.run_lists.values():
            for run in runs:
                if budget is None:
                    budget = run.budget
                elif run.budget != budget:
                    raise Exception("collected runs include multiple run budgets")

        return budget

    @staticmethod
    def from_roots(tasks_roots, domain, suffix = ".runs.csv"):
        """Collect training data by scanning for tasks."""

        task_paths = []

        for tasks_root in tasks_roots:
            task_paths.extend(cargo.files_under(tasks_root, domain.extensions))

        return TrainingData.from_paths(task_paths, domain, suffix)

    @staticmethod
    def from_paths(task_paths, domain, suffix = ".runs.csv"):
        """Collect training data from task paths."""

        training = TrainingData()

        for path in task_paths:
            # load run records
            run_data = numpy.recfromcsv(path + suffix, usemask = True)
            rows = run_data.tolist()

            if run_data.shape == ():
                rows = [rows]

            for (run_solver, run_budget, run_cost, run_succeeded, run_answer) in rows:
                record = RunRecord(run_solver, run_budget, run_cost, run_succeeded)

                training.add_run(path, record)

            # load feature data
            vector = numpy.recfromcsv("{0}.features.csv".format(path)).tolist()

            training.add_feature_vector(path, vector)

        return training

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

