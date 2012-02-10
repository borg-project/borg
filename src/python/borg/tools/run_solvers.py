"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import sys
import csv
import zlib
import base64
import cPickle as pickle
import numpy
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def run_solver_on(suite_path, solver_name, task_path, budget, store_answers, seed = None):
    """Run a solver."""

    if seed is not None:
        borg.statistics.set_prng_seeds(seed)

    suite = borg.load_solvers(suite_path)

    with suite.domain.task_from_path(task_path) as task:
        with borg.accounting() as accountant:
            answer = suite.solvers[solver_name](task)(budget)

        succeeded = suite.domain.is_final(task, answer)

    cost = accountant.total.cpu_seconds

    logger.info(
        "%s %s in %.2f (of %.2f) on %s",
        solver_name,
        "succeeded" if succeeded else "failed",
        cost,
        budget,
        os.path.basename(task_path),
        )

    if store_answers:
        return (task_path, solver_name, budget, cost, succeeded, answer)
    else:
        return (task_path, solver_name, budget, cost, succeeded, None)

@borg.annotations(
    suite_path = ("path to the solvers suite", "positional", None, os.path.abspath),
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    budget = ("per-instance budget", "positional", None, float),
    only_missing = ("only make missing runs", "flag"),
    store_answers = ("store answers to instances", "flag"),
    only_solver = ("only make runs of one solver", "option"),
    runs = ("number of runs", "option", "r", int),
    suffix = ("runs file suffix", "option"),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(
    suite_path,
    tasks_root,
    budget,
    only_missing = False,
    store_answers = False,
    only_solver = None,
    runs = 4,
    suffix = ".runs.csv",
    workers = 0,
    ):
    """Collect solver running-time data."""

    condor.defaults.condor_matching = \
        "InMastodon" \
        " && regexp(\"rhavan-.*\", ParallelSchedulingGroup)" \
        " && (Arch == \"X86_64\")" \
        " && (OpSys == \"LINUX\")" \
        " && (Memory > 1024)"

    def yield_runs():
        suite = borg.load_solvers(suite_path)

        logger.info("scanning paths under %s", tasks_root)

        paths = list(borg.util.files_under(tasks_root, suite.domain.extensions))

        if not paths:
            raise ValueError("no paths found under specified root")

        if only_solver is None:
            solver_names = suite.solvers.keys()
        else:
            solver_names = [only_solver]

        for path in paths:
            run_data = None

            if only_missing and os.path.exists(path + suffix):
                run_data = numpy.recfromcsv(path + suffix, usemask = True)

            for solver_name in solver_names:
                if only_missing and run_data is not None:
                    count = max(0, runs - numpy.sum(run_data.solver == solver_name))
                else:
                    count = runs

                logger.info("scheduling %i run(s) of %s on %s", count, solver_name, os.path.basename(path))

                for _ in xrange(count):
                    seed = numpy.random.randint(sys.maxint)

                    yield (run_solver_on, [suite_path, solver_name, path, budget, store_answers, seed])

    for (task, row) in condor.do(yield_runs(), workers):
        # unpack run outcome
        (cnf_path, solver_name, budget, cost, succeeded, answer) = row

        if answer is None:
            answer_text = None
        else:
            answer_text = base64.b64encode(zlib.compress(pickle.dumps(answer)))

        # write it to disk
        csv_path = cnf_path + suffix
        existed = os.path.exists(csv_path)

        with open(csv_path, "a") as csv_file:
            writer = csv.writer(csv_file)

            if not existed:
                writer.writerow(["solver", "budget", "cost", "succeeded", "answer"])

            writer.writerow([solver_name, budget, cost, succeeded, answer_text])

if __name__ == "__main__":
    borg.script(main)

