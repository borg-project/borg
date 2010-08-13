"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import abstractmethod
from cargo.log   import get_logger
from cargo.flags import (
    Flag,
    Flags,
    )
from borg.rowed  import AbstractRowed

log          = get_logger(__name__)
module_flags = \
    Flags(
        "Solver Configuration",
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

    import numpy

    from numpy import iinfo

    return random.randint(iinfo(numpy.int32).max)

def get_named_solvers(paths = [], flags = {}, use_recycled = False):
    """
    Retrieve a list of named solvers.
    """

    import json

    from os.path      import dirname
    from cargo.io     import expandpath
    from borg.solvers import (
        RecyclingSolver,
        PB_CompetitionSolver,
        SatELitePreprocessor,
        RecyclingPreprocessor,
        SAT_CompetitionSolver,
        )

    flags = module_flags.merged(flags)

    def sat_competition_loader(name, relative, attributes):
        """
        Load a competition solver.
        """

        if use_recycled:
            return RecyclingSolver(name)
        else:
            return SAT_CompetitionSolver(attributes["command"], relative)

    def pb_competition_loader(name, relative, attributes):
        """
        Load a competition solver.
        """

        if use_recycled:
            return RecyclingSolver(name)
        else:
            return PB_CompetitionSolver(attributes["command"], relative)

    def satelite_loader(name, relative, attributes):
        """
        Load a SatELite solver.
        """

        if use_recycled:
            return RecyclingPreprocessor(name)
        else:
            return SatELitePreprocessor(attributes["command"], relative)

    loaders = {
        "sat_competition" : sat_competition_loader,
        "pb_competition"  : pb_competition_loader,
        "satelite"        : satelite_loader,
        }

    def yield_solvers_from(raw_path):
        """
        (Recursively) yield solvers from a solvers file.
        """

        path     = expandpath(raw_path)
        relative = dirname(path)

        with open(path) as file:
            loaded = json.load(file)

        log.note("read named-solvers file: %s", raw_path)

        for (name, attributes) in loaded.get("solvers", {}).items():
            yield (name, loaders[attributes["type"]](name, relative, attributes))

            log.debug("constructed named solver %s", name)

        for included in loaded.get("includes", []):
            for solver in yield_solvers_from(expandpath(included, relative)):
                yield solver

    # build the solvers dictionary
    from itertools import chain

    return dict(chain(*(yield_solvers_from(p) for p in chain(paths, flags.solvers_file))))

class AbstractSolver(AbstractRowed):
    """
    Abstract base for a solver.
    """

    @abstractmethod
    def solve(self, task, budget, random, environment):
        """
        Attempt to solve the specified instance.
        """

    def name(self):
        """
        An arbitrary name for this solver.
        """

        return "anonymous"

class AbstractPreprocessor(AbstractSolver):
    """
    Abstract base for a preprocessor.
    """

    @abstractmethod
    def preprocess(self, task, budget, output_path, random, environment):
        """
        Preprocess an instance.
        """

    @abstractmethod
    def extend(self, task, answer, environment):
        """
        Extend an answer to a preprocessed task back to its parent task.
        """

    @abstractmethod
    def make_task(self, seed, input_task, output_path, environment, row = None):
        """
        Construct an appropriate preprocessed task from its output directory.
        """

class Environment(object):
    """
    Global properties of solver execution.
    """

    def __init__(
        self,
        time_ratio    = 1.0,
        named_solvers = None,
        collections   = {},
        MainSession   = None,
        CacheSession  = None,
        ):
        """
        Initialize.
        """

        self.time_ratio    = time_ratio
        self.named_solvers = named_solvers
        self.collections   = collections
        self.MainSession   = MainSession
        self.CacheSession  = CacheSession

class TypicalEnvironmentFactory(object):
    """
    Build a typical environment.
    """

    def __init__(self, main_url = None, cache_path = None, **kwargs):
        """
        Initialize.
        """

        self._main_url   = main_url
        self._cache_path = cache_path
        self._kwargs     = kwargs

    def __call__(self, engines):
        """
        Build an environment.
        """

        from cargo.sql.alchemy import make_session

        MainSession = make_session(bind = engines.get(self._main_url))

        if self._cache_path is None:
            CacheSession = MainSession
        else:
            from cargo.io import cache_file

            cache_engine = engines.get("sqlite:///%s" % cache_file(self._cache_path))
            CacheSession = make_session(bind = cache_engine)

        return \
            Environment(
                MainSession  = MainSession,
                CacheSession = CacheSession,
                **self._kwargs
                )

