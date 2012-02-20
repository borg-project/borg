"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

@borg.annotations(
    suite_path = ("path to the solvers suite", "positional", None, os.path.abspath),
    solver_name = ("name of solver to run", "positional"),
    instance_path = ("path to problem instance", "positional"),
    specifics = ("other instance parameters (ignored)", "positional"),
    budget = ("run cutoff in seconds", "positional", None, float),
    max_length = ("length cutoff (ignored)", "positional"),
    seed = ("solver seed", "positional", None, int),
    )
def main(
    suite_path,
    solver_name,
    instance_path,
    specifics = None,
    budget = 1e6,
    max_length = None,
    seed = None,
    *args
    ):
    """Make a single solver run for ParamILS."""

    suite = borg.load_solvers(suite_path)

    if seed is not None:
        borg.statistics.set_prng_seeds(seed)

    suite = borg.load_solvers(suite_path)
    solver = suite.solvers[solver_name].with_args(args)

    with suite.domain.task_from_path(instance_path) as task:
        with borg.accounting() as accountant:
            answer = solver(task)(budget)

        succeeded = suite.domain.is_final(task, answer)

    print \
        "Result for ParamILS: {solved}, {run_time}, {run_length}, {best}, {seed}".format(
            solved = "SAT" if succeeded else "TIMEOUT",
            run_time = accountant.total.cpu_seconds,
            run_length = -1,
            best = -1,
            seed = seed,
            )

if __name__ == "__main__":
    borg.script(main)

