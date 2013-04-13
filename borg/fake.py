"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import contextlib
import numpy
import borg

logger = borg.get_logger(__name__, default_level = "DETAIL")

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
            raise Exception("no runs of solver \"{0}\" are recorded".format(self._solver_name))

        run = our_runs[numpy.random.randint(len(our_runs))]

        return FakeSolverProcess(run)

class FakeDomain(object):
    name = "fake"

    def __init__(self, suite):
        """Initialize."""

        self._suite = suite

    @contextlib.contextmanager
    def task_from_path(self, task_path):
        yield task_path

    def compute_features(self, instance):
        """Return static features of an instance."""

        # get features with cost
        with_cost = self._suite.run_data.get_feature_vector(instance)

        borg.get_accountant().charge_cpu(with_cost["cpu_cost"])

        # return the features
        features = dict(with_cost.iteritems())

        del features["cpu_cost"]

        return (features.keys(), features.values())

    def is_final(self, task, answer):
        """Does this answer imply success?"""

        return bool(answer)

    @property
    def extensions(self):
        """Not applicable."""

        raise NotImplementedError()

class FakeSuite(object):
    """Mimic a solver suite using simulated solvers."""

    def __init__(self, run_data):
        """Initialize."""

        self.run_data = run_data
        self.domain = FakeDomain(self)
        self.solvers = dict((k, FakeSolverFactory(k, self.run_data)) for k in self.run_data.solver_names)

