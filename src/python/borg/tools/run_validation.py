"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.run_validation import main

    plac.call(main)

import os.path
import csv
import uuid
import itertools
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
        answer_map = {"": None, "True": True, "False": False}

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

        self.go(budget)

        (solver_id, run_cost, answer, terminated) = self._stm_queue.get()

        assert solver_id == self._solver_id

        self.die()

        return (run_cost, answer)

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

#subsolvers = [
    #"kcnfs-2006",
    #"hybridGM3",
    #"NCVWr",
    #"gnovelty+2",
    #"iPAWS",
    #"adaptg2wsat2009++",
    #"TNM",
    #"march_hi",
    #"FH",
    #]
subsolvers = [
    "pbct-0.1.2-linear",
    "bsolo_pb10-l1",
    "bsolo_pb10-l2",
    "bsolo_pb10-l3",
    "wbo1.4a",
    "wbo1.4b-fixed",
    "clasp-1.3.7",
    "sat4j-pb-v20101225",
    "sat4j-pb-v20101225-cutting",
    ]

def fake_solver(solver_name):
    return lambda *args: FakeSolver(solver_name, *args)

core_solvers = dict(zip(subsolvers, map(fake_solver, subsolvers)))

portfolios = borg.portfolios.named.copy()

def run_validation(name, train_paths, test_paths, budget, split):
    """Make a validation run."""

    solver = portfolios[name](core_solvers, train_paths)
    successes = []

    logger.info("running portfolio %s with per-task budget %.2f", name, budget)

    for test_path in test_paths:
        ## print oracle knowledge, if any
        #(runs,) = borg.portfolios.get_task_run_data([test_path]).values()
        #(oracle_history, oracle_counts, _) = \
            #borg.portfolios.action_rates_from_runs(
                #self._solver_name_index,
                #self._budget_index,
                #runs.tolist(),
                #)
        #true_rates = oracle_history / oracle_counts

        #logger.debug("true rates:\n%s", cargo.pretty_probability_matrix(true_rates))

        # run the portfolio
        with borg.accounting() as accountant:
            answer = solver(test_path, borg.Cost(cpu_seconds = budget))
            cpu_seconds = accountant.total.cpu_seconds

        logger.info("%s answered %s to %s", name, answer, os.path.basename(test_path))

        if answer is not None:
            successes.append(cpu_seconds)

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
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    tests_root = ("optional separate test set", "positional", None, os.path.abspath),
    runs = ("number of runs", "option", "r", int),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(out_path, tasks_root, tests_root = None, runs = 16, workers = 0):
    """Collect validation results."""

    cargo.enable_default_logging()

    cargo.get_logger("borg.portfolios", level = "DETAIL")

    def yield_runs():
        #paths = list(cargo.files_under(tasks_root, ["*.cnf"])) # XXX
        paths = list(cargo.files_under(tasks_root, ["*.opb"]))
        examples = int(round(len(paths) * 0.50))

        logger.info("found %i training tasks", len(paths))

        if tests_root is not None:
            #tests_root_paths = list(cargo.files_under(tests_root, ["*.cnf"]))
            tests_root_paths = list(cargo.files_under(tests_root, ["*.opb"])) # XXX

        for _ in xrange(runs):
            shuffled = sorted(paths, key = lambda _ : numpy.random.rand())
            train_paths = shuffled[:examples]

            if tests_root is None:
                test_paths = shuffled[examples:]
            else:
                test_paths = tests_root_paths

            split = uuid.uuid4()

            for name in portfolios:
                yield (run_validation, [name, train_paths, test_paths, 2600.0, split])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["name", "budget", "cost", "rate", "split"])

        cargo.distribute_or_labor(yield_runs(), workers, lambda _, r: writer.writerows(r))

