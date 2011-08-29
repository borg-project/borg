"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import contextlib
import numpy
import borg
import cargo

logger = cargo.get_logger(__name__, default_level = "DETAIL")

class FakeSolverProcess(object):
    """Provide a solver interface to stored run data."""

    def __init__(self, run):
        """Initialize."""

        self._run = run
        self._elapsed = 0.0
        self._terminated = False

    def run_then_stop(self, budget):
        """Unpause the solver for the specified duration."""

        try:
            return self.run_then_pause(budget)
        finally:
            self.stop()

    def run_then_pause(self, budget):
        """Unpause the solver for the specified duration."""

        assert not self._terminated

        position = self._elapsed + budget

        logger.detail(
            "moving %s run to %.0f of %.0f (from %.0f)",
            self._run.solver,
            position,
            self._run.cost,
            self._elapsed,
            )

        if position >= self._run.cost:
            borg.get_accountant().charge_cpu(self._run.cost - self._elapsed)

            self._elapsed = self._run.cost
            self._terminated = True

            return self._run.success
        else:
            borg.get_accountant().charge_cpu(budget)

            self._elapsed = position

            return None

    def stop(self):
        """Terminate the solver."""

        self._position = None

    @property
    def name(self):
        """Name of the running solver."""

        return self._run.solver

    @property
    def elapsed(self):
        """Number of CPU seconds used by this process (or processes)."""

        return self._elapsed

    @property
    def terminated(self):
        """Has this process terminated?"""

        return self._terminated

class FakeSolverFactory(object):
    def __init__(self, solver_name, runs_data):
        self._solver_name = solver_name
        self._runs_data = runs_data

    def start(self, task):
        """Return a fake solver process."""

        all_runs = self._runs_data.run_lists[task]
        our_runs = filter(lambda run: run.solver == self._solver_name, all_runs)

        if len(our_runs) == 0:
            raise Exception("no candidate runs for fake solver")

        return FakeSolverProcess(cargo.grab(our_runs))

class FakeDomain(object):
    name = "fake"

    def __init__(self, domain):
        """Initialize."""

        self.extensions = [x for x in domain.extensions]

    @contextlib.contextmanager
    def task_from_path(self, task_path):
        yield task_path

    def compute_features(self, task):
        """Read or compute features of an instance."""

        # XXX just pull it out of the training data storage instance
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
        """Does this answer imply success?"""

        return bool(answer)

class FakeSuite(object):
    """Mimic a solver suite, using simulated solvers."""

    def __init__(self, suite, test_paths, suffix):
        """Initialize."""

        self.runs_data = borg.TrainingData.from_paths(test_paths, suite.domain, suffix)
        self.domain = FakeDomain(suite.domain)
        self.solvers = dict((k, FakeSolverFactory(k, self.runs_data)) for k in suite.solvers)

