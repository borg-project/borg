"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.run_solvers import main

    plac.call(main)

import os.path
import csv
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def run_solver_on(domain, solver_name, task_path, budget):
    """Run a solver."""

    with domain.task_from_path(task_path) as task:
        with borg.accounting() as accountant:
            answer = domain.solvers[solver_name](task)(budget)

        succeeded = domain.is_final(task, answer)

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
    domain_name = ("name of the problem domain",),
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    budget = ("per-instance budget", "positional", None, float),
    runs = ("number of runs", "option", "r", int),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(domain_name, tasks_root, budget, runs = 4, workers = 0):
    """Collect solver running-time data."""

    cargo.enable_default_logging()

    def yield_runs():
        domain = borg.get_domain(domain_name)
        paths = list(cargo.files_under(tasks_root, domain.extensions))

        if not paths:
            raise ValueError("no paths found under specified root")

        for _ in xrange(runs):
            for solver_name in domain.solvers:
                for path in paths:
                    yield (run_solver_on, [domain, solver_name, path, budget])

    def collect_run((_, arguments), row):
        (_, cnf_path, _, _) = arguments
        csv_path = cnf_path + ".rtd.csv"
        existed = os.path.exists(csv_path)

        with open(csv_path, "a") as csv_file:
            writer = csv.writer(csv_file)

            if not existed:
                writer.writerow(["solver", "seed", "budget", "cost", "answer"])

            writer.writerow(row)

    cargo.distribute_or_labor(yield_runs(), workers, collect_run)

