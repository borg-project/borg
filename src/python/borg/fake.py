"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import uuid
import random
import contextlib
import multiprocessing
import numpy
import borg

class FakeSolver(object):
    """Provide a solver interface to stored run data."""

    def __init__(self, runs, stm_queue = None, solver_id = None):
        # prepare communication channels
        if stm_queue is None:
            self._stm_queue = multiprocessing.Queue()
        else:
            self._stm_queue = stm_queue

        if solver_id is None:
            self._solver_id = uuid.uuid4()
        else:
            self._solver_id = solver_id

        # gather candidate runs, and select one
        candidate_runs = []

        for run in runs:
            if run.budget >= borg.defaults.minimum_fake_run_budget: # XXX
                candidate_runs.append(run)

        self._run = candidate_runs[random.randrange(len(candidate_runs))]
        self._position = 0.0

    def __call__(self, budget):
        """Unpause the solver, block for some limit, and terminate it."""

        self.unpause_for(budget)

        (solver_id, run_cpu_cost, answer, terminated) = self._stm_queue.get()

        assert solver_id == self._solver_id

        self.stop()

        borg.get_accountant().charge_cpu(run_cpu_cost)

        return answer

    def unpause_for(self, budget):
        """Unpause the solver for the specified duration."""

        assert self._position is not None

        new_position = self._position + budget

        if new_position >= self._run.cost:
            self._stm_queue.put((self._solver_id, self._run.cost - self._position, self._run.success, True))

            self._run_position = None
        else:
            self._stm_queue.put((self._solver_id, budget, None, False))

            self._run_position = new_position

    def stop(self):
        """Terminate the solver."""

        self._position = None

class FakeSolverFactory(object):
    def __init__(self, solver_name, runs_data):
        self._solver_name = solver_name
        self._runs_data = runs_data

    def __call__(self, task, stm_queue = None, solver_id = None):
        all_runs = self._runs_data.get_run_list(task)
        our_runs = filter(lambda run: run.solver == self._solver_name, all_runs)

        assert len(our_runs) > 0

        return FakeSolver(our_runs, stm_queue = stm_queue, solver_id = solver_id)

class FakeDomain(object):
    name = "fake"

    def __init__(self, domain):
        self.extensions = [x for x in domain.extensions]

    @contextlib.contextmanager
    def task_from_path(self, task_path):
        yield task_path

    def compute_features(self, task):
        """Read or compute features of an instance."""

        # grab precomputed feature data
        csv_path = task + ".features.csv"

        assert os.path.exists(csv_path)

        features_array = numpy.recfromcsv(csv_path)
        features = features_array.tolist()
        names = features_array.dtype.names

        # accumulate their cost
        assert names[0] == "cpu_cost"

        borg.get_accountant().charge_cpu(features[0])

        return (names[1:], features[1:])

    def is_final(self, task, answer):
        return answer

class FakeSuite(object):
    """Mimic a solver suite, using simulated solvers."""

    def __init__(self, suite, test_paths, suffix):
        runs_data = borg.storage.TrainingData(test_paths, suite.domain, suffix)

        self.domain = FakeDomain(suite.domain)
        self.solvers = dict((k, FakeSolverFactory(k, runs_data)) for k in suite.solvers)

