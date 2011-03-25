"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.run_validation import main

    plac.call(main)

import os.path
import csv
import uuid
import itertools
import contextlib
import multiprocessing
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

class FakeSolver(object):
    """Provide a standard interface to a solver process."""

    def __init__(self, name, task_path, stm_queue = None, solver_id = None):
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
        runs = []
        answer_map = {"True": True, "False": False}

        with open(task_path + ".rtd.csv") as csv_file:
            reader = csv.reader(csv_file)

            for (solver_name, _, run_budget, run_cost, run_answer) in reader:
                if solver_name == name:
                    run_cost = float(run_cost)
                    run_budget = float(run_budget)

                    if run_budget >= borg.defaults.minimum_fake_run_budget:
                        runs.append((run_cost, answer_map[run_answer]))

        (self._run_cost, self._run_answer) = cargo.grab(runs)
        self._run_position = 0.0

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

        assert self._run_position is not None

        new_position = self._run_position + budget

        if new_position >= self._run_cost:
            self._stm_queue.put((self._solver_id, self._run_cost - self._run_position, self._run_answer, True))

            self._run_position = None
        else:
            self._stm_queue.put((self._solver_id, budget, None, False))

            self._run_position = new_position

    def stop(self):
        """Terminate the solver."""

        self._run_position = None

class FakeSolverFactory(object):
    def __init__(self, solver_name):
        self._solver_name = solver_name

    def __call__(self, task, stm_queue = None, solver_id = None):
        return FakeSolver(self._solver_name, task, stm_queue = stm_queue, solver_id = solver_id)

class FakeDomain(object):
    name = "fake"

    def __init__(self, domain):
        self._real = domain

        self.extensions = [x + ".rtd.csv" for x in domain.extensions]
        self.solvers = dict(zip(self._real.solvers, map(FakeSolverFactory, self._real.solvers)))

    @contextlib.contextmanager
    def task_from_path(self, task_path):
        yield task_path[:-8]

    def compute_features(self, task, cpu_seconds = None):
        """Read or compute features of an instance."""

        # grab precomputed feature data
        csv_path = task + ".features.csv"

        assert os.path.exists(csv_path)

        features_array = numpy.recfromcsv(csv_path)
        features = features_array.tolist()
        names = features_array.dtype.names

        # accumulate their cost
        assert names[0] == "cpu_cost"

        cpu_cost = features[0]

        borg.get_accountant().charge_cpu(cpu_cost)

        # handle timeout logic, and we're done
        if cpu_seconds is not None:
            if cpu_cost >= cpu_seconds:
                return (["cpu_cost"], [cpu_seconds])
            else:
                assert len(names) > 1

        return (names, features)

    def is_final(self, task, answer):
        return answer

def run_validation(name, domain, train_paths, test_paths, budget, split):
    """Make a validation run."""

    solver = borg.portfolios.named[name](domain, train_paths, 50.0, 42)
    successes = []

    logger.info("running portfolio %s with per-task budget %.2f", name, budget)

    for test_path in test_paths:
        with domain.task_from_path(test_path) as test_task:
            cost_budget = borg.Cost(cpu_seconds = budget)

            with borg.accounting() as accountant:
                answer = solver(test_task, cost_budget)

            succeeded = domain.is_final(test_task, answer)
            cpu_cost = accountant.total.cpu_seconds

            if succeeded:
                successes.append(cpu_cost)

            logger.info(
                "%s %s on %s (%.2f CPU s)",
                name,
                "succeeded" if succeeded else "failed",
                os.path.basename(test_path),
                cpu_cost,
                )

    rate = float(len(successes)) / len(test_paths)

    logger.info("method %s had final success rate %.2f", name, rate)

    return \
        zip(
            itertools.repeat(name),
            itertools.repeat(budget),
            sorted(successes),
            numpy.arange(len(successes) + 1.0) / len(test_paths),
            itertools.repeat(split),
            )

@plac.annotations(
    out_path = ("path to results file", "positional", None, os.path.abspath),
    domain_name = ("name of problem domain"),
    budget = ("CPU seconds per instance", "positional", None, float),
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    tests_root = ("optional separate test set", "positional", None, os.path.abspath),
    live = ("don't simulate the domain", "flag", "l"),
    runs = ("number of runs", "option", "r", int),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(out_path, domain_name, budget, tasks_root, tests_root = None, live = False, runs = 16, workers = 0):
    """Collect validation results."""

    cargo.enable_default_logging()

    cargo.get_logger("borg.portfolios", level = "DETAIL")

    def yield_runs():
        # build solvers and train / test sets
        if live:
            domain = borg.get_domain(domain_name)
        else:
            domain = FakeDomain(borg.get_domain(domain_name))

        paths = list(cargo.files_under(tasks_root, domain.extensions))
        examples = int(round(len(paths) * 0.50))

        logger.info("found %i tasks", len(paths))

        if tests_root is not None:
            tests_root_paths = list(cargo.files_under(tests_root, domain.extensions))

        # build validation runs
        for _ in xrange(runs):
            split = uuid.uuid4()
            shuffled = sorted(paths, key = lambda _ : numpy.random.rand())
            train_paths = shuffled[:examples]

            if tests_root is None:
                test_paths = shuffled[examples:]
            else:
                test_paths = tests_root_paths

            for name in borg.portfolios.named:
                yield (run_validation, [name, domain, train_paths, test_paths, budget, split])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["name", "budget", "cost", "rate", "split"])

        cargo.distribute_or_labor(yield_runs(), workers, lambda _, r: writer.writerows(r))

