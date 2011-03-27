"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from borg.tools.solve import main

    plac.call(main)

import sys
import random
import logging
import cPickle as pickle
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

class CompetitionFormatter(logging.Formatter):
    """A concise log formatter for output during competition."""

    def __init__(self):
        logging.Formatter.__init__(self, "%(levelname)s: %(message)s", "%y%m%d%H%M%S")

    def format(self, record):
        """Format the log record."""

        raw = logging.Formatter.format(self, record)

        def yield_lines():
            lines  = raw.splitlines()
            indent = "c " + " " * (len(record.levelname) + 2)

            yield "c " + lines[0]

            for line in lines[1:]:
                yield indent + line

        return "\n".join(yield_lines())

def enable_output():
    """Set up competition-compliant output."""

    # configure the default global level
    cargo.get_logger(level = cargo.defaults.root_log_level)

    # set up output
    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(CompetitionFormatter())
    handler.setLevel(logging.NOTSET)

    logging.root.addHandler(handler)

@plac.annotations(
    solver_path  = ("path to solver pickle"),
    input_path = ("path to instance"),
    seed = ("PRNG seed", "option", None, int),
    budget = ("time limit (CPU or wall)", "option", None, float),
    cores = ("units of execution", "option", None, int),
    quiet = ("be less noisy", "flag", "q"),
    )
def main(solver_path, input_path, seed = 42, budget = 2e6, cores = 1, quiet = False):
    """Solve a problem instance."""

    try:
        # general setup
        enable_output()

        if not quiet:
            cargo.get_logger("borg.solvers", level = "DETAIL")

        numpy.random.seed(seed)
        random.seed(numpy.random.randint(2**31))

        # run the solver
        logger.info("loaded portfolio from %s", solver_path)

        with open(solver_path) as file:
            (domain, solver) = pickle.load(file)

        logger.info("solving %s", input_path)

        with domain.task_from_path(input_path) as task:
            remaining = budget - borg.get_accountant().total.cpu_seconds
            answer = solver(task, borg.Cost(cpu_seconds = remaining), cores)

            return domain.show_answer(task, answer)
    except KeyboardInterrupt:
        print "\nc terminating on SIGINT"

