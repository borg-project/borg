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

class RunData(object):
    """Load and access portfolio training data."""

    def __init__(self):
        """Initialize."""

        self.run_lists = {}
        self.feature_vectors = {}
        self.common_budget = None

    def add_run(self, id_, run):
        """Add a run to these data."""

        runs = self.run_lists.get(id_)

        if runs is None:
            self.run_lists[id_] = [run]
        else:
            runs.append(run)

        if self.common_budget is None:
            self.common_budget = run.budget
        else:
            assert run.budget == self.common_budget

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

    def to_runs_array(self, solver_names):
        """Return run durations as a partially-filled array."""

        S = len(solver_names)
        N = len(self.run_lists)

        # accumulate the success and failure counts
        successes_NS = numpy.zeros((N, S), numpy.intc)
        failures_NS = numpy.zeros((N, S), numpy.intc)
        solver_names_S = list(solver_names)

        for (n, runs) in enumerate(self.run_lists.itervalues()):
            for run in runs:
                s = solver_names_S.index(run.solver)

                if run.success:
                    successes_NS[n, s] += 1
                else:
                    failures_NS[n, s] += 1

        R = numpy.max(successes_NS)

        # fill in run durations
        durations_NSR = numpy.ones((N, S, R), numpy.double) * numpy.nan

        successes_NS[...] = 0

        for (n, runs) in enumerate(self.run_lists.itervalues()):
            for run in runs:
                s = solver_names_S.index(run.solver)
                r = successes_NS[n, s]

                if run.success:
                    durations_NSR[n, s, r] = run.cost
                else:
                    durations_NSR[n, s, r] = -1.0

                successes_NS[n, s] = r + 1

        return (successes_NS, failures_NS, durations_NSR)

    def to_bins_array(self, solver_names, B):
        """Return discretized run duration counts."""

        S = len(solver_names)
        N = len(self.run_lists)
        C = B + 1

        successes_NSB = numpy.zeros((N, S, B), numpy.intc)
        failures_NS = numpy.zeros((N, S), numpy.intc)

        cutoff = self.get_common_budget()
        interval = cutoff / B

        for (n, runs) in enumerate(self.run_lists.values()):
            for run in runs:
                s = solver_names.index(run.solver)

                if run.success and run.cost < cutoff:
                    b = int(run.cost / interval)

                    successes_NSB[n, s, b] += 1
                else:
                    failures_NS[n, s] += 1

        return counts_NSC

    @staticmethod
    def from_roots(tasks_roots, domain, suffix = ".runs.csv"):
        """Collect training data by scanning for tasks."""

        task_paths = []

        for tasks_root in tasks_roots:
            task_paths.extend(cargo.files_under(tasks_root, domain.extensions))

        return RunData.from_paths(task_paths, domain, suffix)

    @staticmethod
    def from_paths(task_paths, domain, suffix = ".runs.csv"):
        """Collect training data from task paths."""

        training = RunData()

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

TrainingData = RunData

