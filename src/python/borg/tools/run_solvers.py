"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.run_solvers import main

    plac.call(main)

import os.path
import csv
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def run_solver_on(solvers_path, solver_name, task_path, budget):
    """Run a solver."""

    bundle = borg.load_solvers(solvers_path)

    with bundle.domain.task_from_path(task_path) as task:
        with borg.accounting() as accountant:
            answer = bundle.solvers[solver_name](task)(budget)

        succeeded = bundle.domain.is_final(task, answer)

    cost = accountant.total.cpu_seconds

    logger.info(
        "%s %s in %.2f (of %.2f) on %s",
        solver_name,
        "succeeded" if succeeded else "failed",
        cost,
        budget,
        os.path.basename(task_path),
        )

    return (solver_name, None, budget, cost, succeeded)

@plac.annotations(
    solvers_path = ("path to the solvers bundle", "positional", None, os.path.abspath),
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    budget = ("per-instance budget", "positional", None, float),
    discard = ("do not record results", "flag", "d"),
    runs = ("number of runs", "option", "r", int),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(solvers_path, tasks_root, budget, discard = False, runs = 4, workers = 0):
    """Collect solver running-time data."""

    cargo.enable_default_logging()

    def yield_runs():
        bundle = borg.load_solvers(solvers_path)
        paths = list(cargo.files_under(tasks_root, bundle.domain.extensions))

        if not paths:
            raise ValueError("no paths found under specified root")

        for _ in xrange(runs):
            for solver_name in bundle.solvers:
                for path in paths:
                    yield (run_solver_on, [solvers_path, solver_name, path, budget])

    def collect_run(task, row):
        if not discard:
            (_, _, cnf_path, _) = task.args
            csv_path = cnf_path + ".rtd.csv"
            existed = os.path.exists(csv_path)

            with open(csv_path, "a") as csv_file:
                writer = csv.writer(csv_file)

                if not existed:
                    writer.writerow(["solver", "seed", "budget", "cost", "answer"])

                writer.writerow(row)

    cargo.do_or_distribute(yield_runs(), workers, collect_run)

