"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.get_features import main

    plac.call(main)

import os.path
import csv
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def features_for_path(domain, task_path):
    logger.info("getting features of %s", os.path.basename(task_path))

    with domain.task_from_path(task_path) as task:
        with borg.accounting() as accountant:
            (names, values) = domain.compute_features(task)

        return (["cpu_cost"] + list(names), [accountant.total.cpu_seconds] + list(values))

@plac.annotations(
    domain_name = ("name of the problem domain",),
    tasks_root = ("path to task files", "positional", None, os.path.abspath),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(domain_name, tasks_root, workers = 0):
    """Collect task features."""

    cargo.enable_default_logging()

    def yield_runs():
        domain = borg.get_domain(domain_name)
        paths = list(cargo.files_under(tasks_root, domain.extensions))

        for path in paths:
            yield (features_for_path, [domain, path])

    def collect_run(task, (names, values)):
        (_, cnf_path) = task.args
        csv_path = cnf_path + ".features.csv"

        with open(csv_path, "w") as csv_file:
            csv.writer(csv_file).writerow(names)
            csv.writer(csv_file).writerow(values)

    cargo.do_or_distribute(yield_runs(), workers, collect_run)

