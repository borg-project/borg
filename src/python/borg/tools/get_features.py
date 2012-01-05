"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def features_for_path(domain, task_path):
    logger.info("getting features of %s", os.path.basename(task_path))

    with domain.task_from_path(task_path) as task:
        with borg.accounting() as accountant:
            (names, values) = domain.compute_features(task)

        return (["cpu_cost"] + list(names), [accountant.total.cpu_seconds] + list(values))

@borg.annotations(
    domain_name = ("suite path, or name of the problem domain",),
    instances_root = ("path to instances files", "positional", None, os.path.abspath),
    workers = ("submit jobs?", "option", "w", int),
    )
def main(domain_name, instances_root, workers = 0):
    """Collect task features."""

    def yield_runs():
        if os.path.exists(domain_name):
            domain = borg.load_solvers(domain_name).domain
        else:
            domain = borg.get_domain(domain_name)

        paths = list(borg.util.files_under(instances_root, domain.extensions))

        for path in paths:
            yield (features_for_path, [domain, path])

    for (task, (names, values)) in condor.do(yield_runs(), workers):
        (_, cnf_path) = task.args
        csv_path = cnf_path + ".features.csv"

        with open(csv_path, "w") as csv_file:
            csv.writer(csv_file).writerow(names)
            csv.writer(csv_file).writerow(values)

if __name__ == "__main__":
    borg.script(main)

