"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.cactus import main

    plac.call(main)

import csv
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

@plac.annotations(
    out_path = ("path to output csv"),
    runs_paths  = ("paths to runs files"),
    )
def main(out_path, *runs_paths):
    """Generate cactus plot data from runs data."""

    # prepare
    cargo.enable_default_logging()

    csv.field_size_limit(1000000000)

    # collect runtimes for each solver
    solver_costs = {}
    solver_names = None

    for runs_path in runs_paths:
        logger.info("fetching runs from %s", runs_path)

        solver_names_here = set()

        with open(runs_path) as runs_file:
            reader = csv.reader(runs_file)

            reader.next()

            for (solver, _, cost, succeeded, _) in reader:
                assert solver not in solver_names_here

                solver_names_here.add(solver)

                runtimes = []

                if solver in solver_costs:
                    runtimes = solver_costs[solver]
                else:
                    solver_costs[solver] = runtimes = []

                if succeeded == "True":
                    runtimes.append(float(cost))

        if solver_names is None:
            solver_names = solver_names_here
        else:
            if solver_names != solver_names_here:
                difference = solver_names.symmetric_difference(solver_names_here)

                raise RuntimeError("solver set mismatch involving: {0}".format(difference))

    # write the cactus data
    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["solver", "solved", "cost"])

        for solver in solver_costs:
            for (i, cost) in enumerate(sorted(solver_costs[solver])):
                writer.writerow([solver, i + 1, cost])

