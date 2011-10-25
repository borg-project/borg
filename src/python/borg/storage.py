"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import random
import itertools
import collections
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

    def __init__(self, solver_names):
        """Initialize."""

        self.solver_names = solver_names
        self.run_lists = {}
        self.feature_vectors = {}
        self.common_budget = None
        self.common_features = None

    def __len__(self):
        """Number of instances for which data are stored."""

        return len(self.run_lists)

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
        assert isinstance(vector, collections.Mapping)

        if self.common_features is None:
            self.common_features = sorted(vector)
        else:
            assert self.common_features == sorted(vector)

        self.feature_vectors[id_] = vector

    def filter(self, *ids):
        """Return a filtered set of run data."""

        data = RunData(self.solver_names)

        for id_ in ids:
            for run in self.run_lists[id_]:
                data.add_run(id_, run)

            data.add_feature_vector(id_, self.feature_vectors[id_])

        data.common_budget = self.common_budget

        return data

    def split(self, fraction):
        """Randomly split the data into two sets of instances."""

        shuffled_ids = sorted(self.ids, key = lambda _: random.random())
        split_size = int(fraction * len(self.ids))
        ids_a = shuffled_ids[:split_size]
        ids_b = shuffled_ids[split_size:]

        return (self.filter(*ids_a), self.filter(*ids_b))

    def masked(self, mask):
        """Return a subset of the instances."""

        return self.filter(*(id_ for (id_, m) in zip(self.ids, mask) if m))

    def collect(self, counts):
        """Get a systematic subset of the data."""

        sampled = RunData(self.solver_names)
        iter_count = itertools.cycle(counts)

        for id_ in self.ids:
            count = next(iter_count)

            shuffled_runs = sorted(self.run_lists[id_], key = lambda _: random.random())

            for i in xrange(count):
                sampled.add_run(id_, shuffled_runs[i])

            sampled.add_feature_vector(id_, self.feature_vectors[id_])

        sampled.common_budget = self.common_budget

        return sampled

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

    def to_features_array(self):
        """Retrieve feature values in an array."""

        assert set(self.feature_vectors) == set(self.run_lists)

        N = len(self.feature_vectors)
        F = len(self.common_features)

        feature_values_NF = numpy.empty((N, F), numpy.double)

        for (n, instance_id) in enumerate(sorted(self.feature_vectors)):
            features = self.feature_vectors[instance_id]

            for (f, name) in enumerate(self.common_features):
                feature_values_NF[n, f] = features[name]

        return feature_values_NF

    def to_runs_array(self, solver_names):
        """Return run durations as a partially-filled array."""

        S = len(solver_names)
        N = len(self.run_lists)

        # accumulate the success and failure counts
        successes_NS = numpy.zeros((N, S), numpy.intc)
        failures_NS = numpy.zeros((N, S), numpy.intc)
        solver_names_S = list(solver_names)
        instance_ids = sorted(self.run_lists)

        for (n, instance_id) in enumerate(instance_ids):
            runs = self.run_lists[instance_id]

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

        for (n, instance_id) in enumerate(instance_ids):
            runs = self.run_lists[instance_id]

            for run in runs:
                s = solver_names_S.index(run.solver)

                if run.success:
                    r = successes_NS[n, s]

                    durations_NSR[n, s, r] = run.cost

                    successes_NS[n, s] = r + 1

        return (successes_NS, failures_NS, durations_NSR)

    def to_bins_array(self, solver_names, B):
        """Return discretized run duration counts."""

        S = len(solver_names)
        N = len(self.run_lists)
        C = B + 1

        solver_name_index = list(solver_names)
        outcomes_NSC = numpy.zeros((N, S, C), numpy.intc)
        cutoff = self.get_common_budget()
        interval = cutoff / B

        for (n, instance_id) in enumerate(sorted(self.run_lists)):
            runs = self.run_lists[instance_id]

            for run in runs:
                s = solver_name_index.index(run.solver)

                if run.success and run.cost < cutoff:
                    b = int(run.cost / interval)

                    outcomes_NSC[n, s, b] += 1
                else:
                    outcomes_NSC[n, s, B] += 1

        return outcomes_NSC

    @property
    def ids(self):
        """All associated instance ids."""

        return self.run_lists.keys()

    @staticmethod
    def from_roots(solver_names, tasks_roots, domain, suffix = ".runs.csv"):
        """Collect run data by scanning for tasks."""

        task_paths = []

        for tasks_root in tasks_roots:
            task_paths.extend(cargo.files_under(tasks_root, domain.extensions))

        return RunData.from_paths(solver_names, task_paths, domain, suffix)

    @staticmethod
    def from_paths(solver_names, task_paths, domain, suffix = ".runs.csv"):
        """Collect run data from task paths."""

        training = RunData(solver_names)

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
            feature_records = numpy.recfromcsv("{0}.features.csv".format(path))
            feature_dict = dict(zip(feature_records.dtype.names, feature_records.tolist()))

            del feature_dict["cpu_cost"]
            #feature_dict = {"nvars": feature_dict["nvars"]}

            training.add_feature_vector(path, feature_dict)

        return training

    @staticmethod
    def from_bundle(bundle_path):
        """Collect run data from two CSV files."""

        run_data = RunData(None)

        # load runs
        runs_csv_path = os.path.join(bundle_path, "all_runs.csv.gz")

        logger.info("reading run data from %s", runs_csv_path)

        solver_names = set()

        with cargo.openz(runs_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)

            columns = csv_reader.next()

            if columns[:5] != ["instance", "solver", "budget", "cost", "succeeded"]:
                raise Exception("unexpected columns in run data CSV file")

            for (instance, solver, budget_str, cost_str, succeeded_str) in csv_reader:
                run_data.add_run(
                    instance,
                    RunRecord(
                        solver,
                        float(budget_str),
                        float(cost_str),
                        succeeded_str.lower() == "true",
                        ),
                    )
                solver_names.add(solver)

        run_data.solver_names = sorted(solver_names)

        # load features
        features_csv_path = os.path.join(bundle_path, "all_features.csv.gz")

        logger.info("reading feature data from %s", features_csv_path)

        with cargo.openz(features_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)

            columns = csv_reader.next()

            if columns[0] != "instance":
                raise Exception("unexpected columns in features CSV file")

            for row in csv_reader:
                feature_dict = dict(zip(columns[1:], map(float, row[1:])))

                del feature_dict["cpu_cost"]

                run_data.add_feature_vector(row[0], feature_dict)

        assert set(run_data.run_lists) == set(run_data.feature_vectors)

        return run_data

TrainingData = RunData

