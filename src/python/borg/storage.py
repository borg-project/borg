"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import itertools
import collections
import numpy
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

class RunRecord(object):
    """Record of a solver run."""

    def __init__(self, solver, budget, cost, success):
        """Initialize."""

        self.solver = solver
        self.budget = budget
        self.cost = cost
        self.success = success

    def __str__(self):
        return str((self.solver, self.budget, self.cost, self.success))

    def __repr__(self):
        return repr((self.solver, self.budget, self.cost, self.success))

class RunData(object):
    """Load and access portfolio training data."""

    def __init__(self, solver_names, common_budget = None):
        """Initialize."""

        self.solver_names = solver_names
        self.run_lists = {}
        self.feature_vectors = {}
        self.common_budget = common_budget
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

    def add_runs(self, pairs):
        """Add runs to these data."""

        for (id_, run) in pairs:
            self.add_run(id_, run)

    def add_feature_vector(self, id_, vector):
        """Add a feature vector to these data."""

        assert id_ not in self.feature_vectors
        assert isinstance(vector, collections.Mapping)

        names = [k for k in vector if k != "cpu_cost"]

        if self.common_features is None:
            self.common_features = sorted(names)
        else:
            assert self.common_features == sorted(names)

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

    def filter_features(self, names):
        """Return a set of run data with only the specified features."""

        data = RunData(self.solver_names, self.common_budget)

        data.run_lists = self.run_lists

        for (id_, old_vector) in self.feature_vectors.iteritems():
            new_vector = dict((k, old_vector[k]) for k in names)

            data.add_feature_vector(id_, new_vector)

        return data

    def masked(self, mask):
        """Return a subset of the instances."""

        return self.filter(*(id_ for (id_, m) in zip(self.ids, mask) if m))

    def only_successful(self):
        """Return only instances on which some solver succeeded."""

        data = RunData(self.solver_names)

        for (id_, run_list) in self.run_lists.iteritems():
            if any(run.success for run in run_list):
                for run in run_list:
                    data.add_run(id_, run)

                data.add_feature_vector(id_, self.feature_vectors[id_])

        data.common_budget = self.common_budget

        return data

    def only_nontrivial(self, threshold = 1.0):
        """Return only instances on which some solver succeeded."""

        data = RunData(self.solver_names)

        for (id_, run_list) in self.run_lists.iteritems():
            if any(not run.success or run.cost > threshold for run in run_list):
                for run in run_list:
                    data.add_run(id_, run)

                data.add_feature_vector(id_, self.feature_vectors[id_])

        data.common_budget = self.common_budget

        return data

    def only_nonempty(self):
        """Return only instances on which some solver succeeded."""

        data = RunData(self.solver_names)

        for (id_, run_list) in self.run_lists.iteritems():
            if len(run_list) > 0:
                for run in run_list:
                    data.add_run(id_, run)

                data.add_feature_vector(id_, self.feature_vectors[id_])

        data.common_budget = self.common_budget

        return data

    def collect_systematic(self, counts):
        """Get a systematic subset of the data."""

        sampled = RunData(self.solver_names, common_budget = self.common_budget)
        iter_count = itertools.cycle(counts)

        for id_ in sorted(self.ids, key = lambda _: numpy.random.rand()):
            count = next(iter_count)

            for solver in self.solver_names:
                runs = sorted(self.runs_on(id_, solver), key = lambda _: numpy.random.rand())

                assert len(runs) >= count

                sampled.add_runs((id_, run) for run in runs[:count])

            sampled.add_feature_vector(id_, self.feature_vectors[id_])

        return sampled

    def collect_independent(self, counts):
        """Get independent subsets of the data."""

        sampled = RunData(self.solver_names, common_budget = self.common_budget)

        for solver in self.solver_names:
            iter_count = itertools.cycle(counts)

            for id_ in sorted(self.ids, key = lambda _: numpy.random.rand()):
                count = next(iter_count)
                runs = sorted(self.runs_on(id_, solver), key = lambda _: numpy.random.rand())

                sampled.add_runs((id_, run) for run in runs[:count])

                if id_ not in sampled.feature_vectors:
                    sampled.add_feature_vector(id_, self.feature_vectors[id_])

        return sampled

    def runs_on(self, id_, solver):
        """Retrieve runs made by a solver on an instance."""

        for run in self.run_lists[id_]:
            if run.solver == solver:
                yield run

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

    def get_run_count(self):
        """Return the number of runs stored."""

        return sum(map(len, self.run_lists.values()))

    def to_features_array(self):
        """Retrieve feature values in an array."""

        assert set(self.feature_vectors) == set(self.run_lists)

        N = len(self.ids)
        F = len(self.common_features)

        feature_values_NF = numpy.empty((N, F), numpy.double)

        for (n, instance_id) in enumerate(sorted(self.ids)):
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

    def to_times_arrays(self):
        """Return run durations as per-solver arrays."""

        S = len(self.solver_names)
        N = len(self.run_lists)

        times_lists = [[] for _ in xrange(S)]
        ns_lists = [[] for _ in xrange(S)]
        failures_NS = numpy.zeros((N, S), numpy.intc)
        instance_ids = sorted(self.run_lists)

        for (n, instance_id) in enumerate(instance_ids):
            runs = self.run_lists[instance_id]

            for run in runs:
                s = self.solver_names.index(run.solver)

                if run.success:
                    times_lists[s].append(run.cost)
                    ns_lists[s].append(n)
                else:
                    failures_NS[n, s] += 1

        times_arrays = map(numpy.array, times_lists)
        ns_arrays = map(numpy.array, ns_lists)

        return (times_arrays, ns_arrays, failures_NS)

    def to_bins_array(self, solver_names, B, cutoff = None):
        """Return discretized run duration counts."""

        if cutoff is None:
            cutoff = self.get_common_budget()

        S = len(solver_names)
        N = len(self.run_lists)
        C = B + 1

        solver_name_index = list(solver_names)
        outcomes_NSC = numpy.zeros((N, S, C), numpy.intc)
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
            task_paths.extend(borg.util.files_under(tasks_root, domain.extensions))

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

        with borg.util.openz(runs_csv_path) as csv_file:
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

        with borg.util.openz(features_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)

            try:
                columns = csv_reader.next()
            except StopIteration:
                pass
            else:
                if columns[0] != "instance":
                    raise Exception("unexpected columns in features CSV file")

                for row in csv_reader:
                    feature_dict = dict(zip(columns[1:], map(float, row[1:])))

                    run_data.add_feature_vector(row[0], feature_dict)

                assert set(run_data.run_lists) == set(run_data.feature_vectors)

        return run_data

TrainingData = RunData

