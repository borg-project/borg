"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from borg.tools.get_features import main

    plac.call(main)

import os.path
import csv
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def features_for_path(domain, task_path, cpu_seconds):
    logger.info("getting features of %s", os.path.basename(task_path))

    with domain.task_from_path(task_path) as task:
        return domain.compute_features(task, cpu_seconds)

@plac.annotations(
    domain_name = ("name of the problem domain",),
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    cpu_seconds = ("CPU time limit", "positional", None, float),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(domain_name, tasks_root, cpu_seconds = 1e8, workers = 0):
    """Collect task features."""

    cargo.enable_default_logging()

    def yield_runs():
        domain = borg.get_domain(domain_name)
        paths = list(cargo.files_under(tasks_root, domain.extensions))

        for path in paths:
            yield (features_for_path, [domain, path, cpu_seconds])

    def collect_run((_, arguments), (names, values)):
        (_, cnf_path, _) = arguments
        csv_path = cnf_path + ".features.csv"

        with open(csv_path, "w") as csv_file:
            csv.writer(csv_file).writerow(names)
            csv.writer(csv_file).writerow(values)

    cargo.distribute_or_labor(yield_runs(), workers, collect_run)

