"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import abc
import collections
import borg

from cargo.log  import get_logger
from borg.rowed import AbstractRowed

logger = get_logger(__name__)

def get_random_seed(random):
    """
    Return a random solver seed.
    """

    import numpy

    from numpy import iinfo

    return random.randint(iinfo(numpy.int32).max)

def get_named_solvers(paths = borg.defaults.solver_lists, use_recycled = False):
    """
    Retrieve a list of named solvers.
    """

    from os.path      import dirname
    from cargo.io     import expandpath
    from borg.solvers import (
        RecyclingSolver,
        RecyclingPreprocessor,
        )

    def sat_competition_loader(name, relative, attributes):
        """
        Load a competition solver.
        """

        from borg.solvers import SAT_CompetitionSolver

        if use_recycled:
            return RecyclingSolver(name)
        else:
            return SAT_CompetitionSolver(attributes["command"], relative)

    def pb_competition_loader(name, relative, attributes):
        """
        Load a competition solver.
        """

        from borg.solvers import PB_CompetitionSolver

        if use_recycled:
            return RecyclingSolver(name)
        else:
            return PB_CompetitionSolver(attributes["command"], relative)

    def satelite_loader(name, relative, attributes):
        """
        Load a SatELite solver.
        """

        from borg.solvers import SatELitePreprocessor

        if use_recycled:
            return RecyclingPreprocessor(name)
        else:
            return SatELitePreprocessor(attributes["command"], relative)

    def zefram_loader(name, relative, attributes):
        """
        Load a Zefram solver.
        """

        from borg.solvers import ZeframSolver

        if use_recycled:
            return RecyclingPreprocessor(name)
        else:
            return ZeframSolver(attributes["command"], relative)

    loaders = {
        "sat_competition" : sat_competition_loader,
        "pb_competition"  : pb_competition_loader,
        "satelite"        : satelite_loader,
        "zefram"          : zefram_loader,
        }

    def yield_solvers_from(raw_path):
        """
        (Recursively) yield solvers from a solvers file.
        """

        from cargo.json import load_json

        path     = expandpath(raw_path)
        relative = dirname(path)
        loaded   = load_json(path)

        logger.note("read named-solvers file: %s", raw_path)

        for (name, attributes) in loaded.get("solvers", {}).items():
            yield (name, loaders[attributes["type"]](name, relative, attributes))

            logger.debug("constructed named solver %s", name)

        for included in loaded.get("includes", []):
            for solver in yield_solvers_from(expandpath(included, relative)):
                yield solver

    # build the solvers dictionary
    from itertools import chain

    return dict(chain(*(yield_solvers_from(p) for p in paths)))

class AbstractSolver(AbstractRowed):
    """
    Abstract base for a solver.
    """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def preprocess(self, task, budget, output_path, random, environment):
        """
        Preprocess an instance.
        """

    @abc.abstractmethod
    def extend(self, task, answer, environment):
        """
        Extend an answer to a preprocessed task back to its parent task.
        """

    @abc.abstractmethod
    def make_task(self, seed, input_task, output_path, environment, row = None):
        """
        Construct an appropriate preprocessed task from its output directory.
        """

AttemptCached = \
    collections.namedtuple(
        "AttemptCached",
        ["budget", "cost", "answer_uuid"],
        )

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

        # XXX build map: (solver, task) -> [(budget, outcome), ...]
        # XXX where outcomes are ordered by budget
        # XXX *forget* about stdout and certificate---if we truly care, we can store
        # XXX the associated primary key and pull them off disk on demand

        with self.CacheSession() as session:
            self.attempts_cache = self._build_outcomes_map(session)

    def _build_outcomes_map(self, session):
        """
        Store observed solver outcomes in memory.
        """

        import sqlalchemy as sql

        attempt_rows = \
            session.execute(
                sql.select(
                    [
                        borg.AttemptRow.budget,
                        borg.AttemptRow.cost,
                        borg.AttemptRow.task_uuid,
                        borg.AttemptRow.answer_uuid,
                        borg.RunAttemptRow.solver_name,
                        ],
                    borg.AttemptRow.uuid == borg.RunAttemptRow.__table__.c.uuid,
                    ),
                )

        cache = {}
        i = 0

        for (i, row) in enumerate(attempt_rows):
            key = (row.solver_name, row.task_uuid)
            value = AttemptCached(row.budget, row.cost, row.answer_uuid)

            if key in cache:
                cache[key].append(value)
            else:
                cache[key] = [value]

        logger.detail("cached %i solver attempt rows", i + 1)

        return cache

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

    def __call__(self, engines, copy_cache = False):
        """
        Build an environment.
        """

        from cargo.sql.alchemy import make_session

        MainSession = make_session(bind = engines.get(self._main_url))

        if self._cache_path is None:
            CacheSession = MainSession
        else:
            if copy_cache:
                from cargo.io import cache_file

                cache_engine = engines.get("sqlite:///%s" % cache_file(self._cache_path))
            else:
                cache_engine = engines.get("sqlite:///%s" % self._cache_path)

            CacheSession = make_session(bind = cache_engine)

        return \
            Environment(
                MainSession  = MainSession,
                CacheSession = CacheSession,
                **self._kwargs
                )

