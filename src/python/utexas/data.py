"""
utexas/data.py

Storage and retrieval of research data.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.data import main

    raise SystemExit(main())

import os.path
import logging

from uuid import (
    UUID,
    uuid4,
    uuid5,
    )
from socket import getfqdn
from datetime import timedelta
from itertools import (
    izip,
    chain,
    product,
    )
from sqlalchemy import (
    Float,
    Column,
    Binary,
    String,
    Integer,
    Boolean,
    ForeignKey,
    UnicodeText,
    )
from sqlalchemy.orm import (
    relation,
    sessionmaker,
    )
from sqlalchemy.orm.exc import NoResultFound
from cargo.io import (
    openz,
    files_under,
    hash_yielded_bytes,
    )
from sqlalchemy.ext.declarative import declarative_base
from cargo.log import get_logger
from cargo.sql.alchemy import (
    SQL_UUID,
    SQL_JSON,
    SQL_List,
    SQL_Engines,
    UTC_DateTime,
    SQL_TimeDelta,
    )
from cargo.flags import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.temporal import utc_now

log             = get_logger(__name__)
DatumBase       = declarative_base()
ResearchSession = sessionmaker()
module_flags    = \
    Flags(
        "Research Data Storage",
        Flag(
            "--research-database",
            default = "sqlite:///:memory:",
            metavar = "DATABASE",
            help    = "use research DATABASE by default [%default]",
            ),
        )

class Task(DatumBase):
    """
    One task.
    """

    __tablename__ = "tasks"

    uuid = Column(SQL_UUID, primary_key = True, default = uuid4)
    type = Column(String)

    __mapper_args__ = {"polymorphic_on": type}

class SAT_Task(Task):
    """
    One satisfiability task in DIMACS CNF format.
    """

    __tablename__   = "sat_tasks"
    __mapper_args__ = {"polymorphic_identity": "cnf_sat"}

    uuid = Column(SQL_UUID, ForeignKey(Task.uuid), primary_key = True, default = uuid4)
    hash = Column(Binary(length = 64))

    TASK_NAMESPACE = UUID("8e67a81a-717c-4206-8831-6007bc8f111f")

class TaskDescription(DatumBase):
    """
    Place a task in the context of a collection.
    """

    __tablename__ = "task_descriptions"

    uuid       = Column(SQL_UUID, primary_key = True, default = uuid4)
    task_uuid  = Column(SQL_UUID, ForeignKey("tasks.uuid"), nullable = False)
    name       = Column(String)
    collection = Column(String)

class Solver(DatumBase):
    """
    Some solver for some domain.
    """

    __tablename__ = "sat_solvers"

#     uuid = Column(SQL_UUID, primary_key = True, default = uuid4)
    name = Column(String, primary_key = True)
    type = Column(String)

class SAT_SolverRun(DatumBase):
    """
    Information about one run of a solver on a SAT task.
    """

    __tablename__ = "sat_solver_runs"

    uuid        = Column(SQL_UUID, primary_key = True, default = uuid4)
    task_uuid   = Column(SQL_UUID, ForeignKey("sat_tasks.uuid"), nullable = False)
    solver_name = Column(String, ForeignKey("sat_solvers.name"), nullable = False)
    outcome     = Column(Boolean)
    started     = Column(UTC_DateTime)
    elapsed     = Column(SQL_TimeDelta)
    cutoff      = Column(SQL_TimeDelta)
    fqdn        = Column(String)
    seed        = Column(Integer)
    exit_code   = Column(Integer)
    output      = Column(UnicodeText)

    task          = relation(SAT_Task)
    solver        = relation(Solver)

    @staticmethod
    def starting_now(*args, **kwargs):
        """
        Return a partially-initialized run starting now.
        """

        assert "started" not in kwargs
        assert "fqdn" not in kwargs

        run = SAT_SolverRun(*args, **kwargs)

        run.started = utc_now()
        run.fqdn    = getfqdn()

        return run

class PerplexityDatum(DatumBase):
    """
    Datum from a perplexity calculation.
    """

    __tablename__ = "perplexity_data"

    uuid            = Column(SQL_UUID, primary_key = True, default = uuid4)
    model_name      = Column(String)
    set_name        = Column(String)
    ntasks_train    = Column(Integer)
    ntasks_test     = Column(Integer)
    nrestarts_train = Column(Integer)
    nrestarts_test  = Column(Integer)
    perplexity      = Column(Float)
    ncomponents     = Column(Integer)

class PortfolioScoreWorld(DatumBase):
    """
    World from a (set of) portfolio tests.
    """

    __tablename__ = "portfolio_score_worlds"

    uuid     = Column(SQL_UUID, primary_key = True, default = uuid4)
    ntrain   = Column(Integer)
    ntest    = Column(Integer)
    limit    = Column(SQL_TimeDelta)
    prefix   = Column(String)
    discount = Column(Float)
    tags     = Column(SQL_List(String))

class PortfolioScore(DatumBase):
    """
    Result of a portfolio test.
    """

    __tablename__ = "portfolio_scores"

    # columns
    uuid         = Column(SQL_UUID, primary_key = True, default = uuid4)
    world_uuid   = Column(SQL_UUID, ForeignKey("portfolio_score_worlds.uuid"))
    model_name   = Column(String)
    planner_name = Column(String)
    components   = Column(Integer)
    solved       = Column(Integer)
    spent        = Column(SQL_TimeDelta)
    utility      = Column(Float)
    tags         = Column(SQL_List(String))

    # relations
    world = relation(PortfolioScoreWorld)

class PortfolioActionCount(DatumBase):
    """
    Action invocation count in a portfolio experiment.
    """

    __tablename__ = "portfolio_action_counts"

    # columns
    uuid        = Column(SQL_UUID, primary_key = True, default = uuid4)
    score_uuid  = Column(SQL_UUID, ForeignKey("portfolio_scores.uuid"))
    solver_name = Column(String, ForeignKey("sat_solvers.name"))
    duration    = Column(SQL_TimeDelta)
    invocations = Column(Integer)

    # relations
    score  = relation(PortfolioScore)
    solver = relation(Solver)

def research_connect(engines = SQL_Engines.default, flags = module_flags.given):
    """
    Connect to research data storage.
    """

    flags  = module_flags.merged(flags)
    engine = engines.get(flags.research_database)

    DatumBase.metadata.create_all(engine)

    return engine

@with_flags_parsed()
def main(positional):
    """
    Create core database metadata.
    """

    research_connect()

