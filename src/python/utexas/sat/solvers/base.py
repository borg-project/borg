"""
utexas/sat/solvers/base.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import (
    abstractmethod,
    abstractproperty,
    )
from cargo.log   import get_logger
from cargo.sugar import ABC
from cargo.flags import (
    Flag,
    Flags,
    )

log          = get_logger(__name__)
module_flags = \
    Flags(
        "SAT Solver Configuration",
        Flag(
            "--solvers-file",
            default = [],
            action  = "append",
            metavar = "FILE",
            help    = "read solver descriptions from FILE [%default]",
            ),
        )

def get_random_seed(random):
    """
    Return a random solver seed.
    """

    return random.randint(2**31 - 1)

def get_named_solvers(paths = [], flags = {}):
    """
    Retrieve a list of named solvers.
    """

    import json

    from os.path  import dirname
    from cargo.io import expandpath

    flags = module_flags.merged(flags)

    def yield_solvers_from(raw_path):
        """
        (Recursively) yield solvers from a solvers file.
        """

        path     = expandpath(raw_path)
        relative = dirname(path)

        with open(path) as file:
            loaded = json.load(file)

        for (name, attributes) in loaded.get("solvers", {}).items():
            yield (
                name,
                SAT_CompetitionSolver(
                    attributes["command"],
                    solvers_home = relative,
                    name         = name,
                    ),
                )

        for included in loaded.get("includes", []):
            for solver in yield_solvers_from(expandpath(included, relative)):
                yield solver

    # build the solvers dictionary
    from itertools import chain

    return dict(chain(*(yield_solvers_from(p) for p in chain(paths, flags.solvers_file))))

class SolverError(RuntimeError):
    """
    The solver failed in an unexpected way.
    """

class SAT_Result(object):
    """
    Minimal outcome of a SAT solver.
    """

    @abstractproperty
    def satisfiable(self):
        """
        Did the solver report the instance satisfiable?
        """

    @abstractproperty
    def certificate(self):
        """
        Certificate of satisfiability, if any.
        """

class SAT_Solver(ABC):
    """
    A solver for SAT.
    """

    @abstractmethod
    def solve(self, input_path, cutoff = None, seed = None):
        """
        Attempt to solve the specified instance; return the outcome.
        """

