"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac
import sys
import logging
import cPickle as pickle
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

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
    borg.get_logger(level = borg.defaults.root_log_level)

    # set up output
    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(CompetitionFormatter())
    handler.setLevel(logging.NOTSET)

    logging.root.addHandler(handler)

@plac.annotations(
    model_path  = ("path to trained model pickle"),
    solvers_path  = ("path to solvers bundle"),
    input_path = ("path to instance"),
    seed = ("PRNG seed", "option", None, int),
    budget = ("time limit (CPU or wall)", "option", None, float),
    cores = ("units of execution", "option", None, int),
    speed = ("machine calibration ratio", "option", "s", float),
    quiet = ("be less noisy", "flag", "q"),
    )
def main(
    model_path,
    solvers_path,
    input_path,
    seed = 42,
    budget = 3600.0,
    cores = 1,
    speed = borg.defaults.machine_speed,
    quiet = False
    ):
    """Solve a problem instance."""

    # XXX hackish
    borg.defaults.machine_speed = speed

    try:
        # general setup
        enable_output()

        if not quiet:
            borg.get_logger("borg.solvers", level = "DETAIL")

        borg.statistics.set_prng_seeds(seed)

        # run the solver
        bundle = borg.load_solvers(solvers_path)

        logger.info("loaded portfolio model from %s", model_path)

        with open(model_path) as file:
            portfolio = pickle.load(file)

        logger.info("solving %s", input_path)

        with bundle.domain.task_from_path(input_path) as task:
            remaining = budget - borg.get_accountant().total.cpu_seconds
            answer = portfolio(task, bundle, borg.Cost(cpu_seconds = remaining), cores)

            return bundle.domain.show_answer(task, answer)
    except KeyboardInterrupt:
        print "\nc terminating on SIGINT"

if __name__ == "__main__":
    plac.call(main)

