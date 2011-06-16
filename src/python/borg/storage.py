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

    def __init__(self, tasks_roots, domain):
        """Initialize."""

        # scan for CSV files
        train_paths = []

        for tasks_root in tasks_roots:
            train_paths.extend(cargo.files_under(tasks_root, domain.extensions))

        logger.info("using %i tasks for training", len(train_paths))

        # fetch training data from each file
        self._run_lists = {}
        self._feature_vectors = {}

        for path in train_paths:
            # load run records
            run_data = numpy.recfromcsv("{0}.rtd.csv".format(path), usemask = True)
            run_list = []

            for (run_solver, _, run_budget, run_cost, run_answer) in run_data.tolist():
                record = RunRecord(run_solver, run_budget, run_cost, run_answer is not None)

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

