"""
utexas/quest/acridid/core.py

Common acridid code.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import socket
import logging

from uuid import (
    UUID,
    uuid4,
    uuid5,
    )
from datetime import timedelta
from itertools import (
    izip,
    chain,
    product,
    )
from sqlalchemy import (
    Column,
    Binary,
    String,
    Integer,
    Boolean,
    Interval,
    ForeignKey,
    )
from sqlalchemy.orm import relation
from sqlalchemy.orm.exc import NoResultFound
from cargo.io import (
    hash_file,
    files_under,
    )
from cargo.log import get_logger
from cargo.sql.alchemy import (
    SQL_Base,
    SQL_UUID,
    SQL_JSON,
    SQL_List,
    SQL_Session,
    UTC_DateTime,
    )
from cargo.temporal import utc_now

log = get_logger(__name__, level = None)

class SAT_Task(SQL_Base):
    """
    One satisfiability task in DIMACS CNF format.
    """

    __tablename__ = "sat_tasks"

    uuid = Column(SQL_UUID, primary_key = True, default = uuid4)
    name = Column(String)
    hash = Column(Binary(length = 64))
    path = Column(String)

    @staticmethod
    def from_file(root, path):
        """
        Describe an existing file.
        """

        # hash and retrieve the task
        (_, hash)     = hash_file(path, "sha512")
        relative_path = os.path.relpath(path, root)
        task          = SAT_Task(path = relative_path, name = os.path.basename(path), hash = hash)

        return task

class SAT_SolverDescription(SQL_Base):
    """
    Information about a SAT solver.
    """

    __tablename__ = "sat_solvers"

    name = Column(String, primary_key = True)
    path = Column(String)

class SAT_SolverConfiguration(SQL_Base):
    """
    Configuration of a SAT solver.
    """

    __tablename__ = "sat_solver_configurations"

    uuid        = Column(SQL_UUID, primary_key = True, default = uuid4)
    name        = Column(String)
    solver_name = Column(String, ForeignKey("sat_solvers.name"))

    __mapper_args__ = {"polymorphic_on": solver_name}

class ArgoSAT_Configuration(SAT_SolverConfiguration):
    """
    Configuration of ArgoSAT.
    """

    __tablename__   = "argosat_configurations"
    __mapper_args__ = {"polymorphic_identity": "argosat"}

    uuid = Column(
        SQL_UUID,
        ForeignKey("sat_solver_configurations.uuid"),
        default     = uuid4,
        primary_key = True,
        )
    variable_selection = Column(SQL_List(String))
    polarity_selection = Column(SQL_List(String))
    restart_scheduling = Column(SQL_List(String))

    NAMED_CONFIGURATION_NAMESPACE = UUID("72f8c280c63f4d81a00473850a34b710")
    VARIABLE_SELECTION_STRATEGIES = {
        "r":   ("random",),
        "m":   ("minisat", "init", "1.0", "1.052"),
        "r+m": ("random", "0.05", "minisat", "init", "1.0", "1.052"),
        }
    POLARITY_SELECTION_STRATEGIES = {
        "t": ("true",),
        "f": ("false",),
        "r": ("random", "0.5"),
        }
    RESTART_SCHEDULING_STRATEGIES = {
        "n": ("no_restart",),
        }

    @staticmethod
    def from_names(vss, pss, rss):
        name = "v:%s,p:%s,r:%s" % (vss, pss, rss)

        if vss is None:
            vss_arguments = None
        else:
            vss_arguments = ArgoSAT_Configuration.VARIABLE_SELECTION_STRATEGIES[vss]

        if pss is None:
            pss_arguments = None
        else:
            pss_arguments = ArgoSAT_Configuration.POLARITY_SELECTION_STRATEGIES[pss]

        if rss is None:
            rss_arguments = None
        else:
            rss_arguments = ArgoSAT_Configuration.RESTART_SCHEDULING_STRATEGIES[rss]

        return \
            ArgoSAT_Configuration(
                uuid = uuid5(
                    ArgoSAT_Configuration.NAMED_CONFIGURATION_NAMESPACE,
                    name
                    ),
                name               = name,
                variable_selection = vss_arguments,
                polarity_selection = pss_arguments,
                restart_scheduling = rss_arguments,
                )

    @staticmethod
    def yield_all():
        s = \
            product(
                ArgoSAT_Configuration.VARIABLE_SELECTION_STRATEGIES.iterkeys(),
                ArgoSAT_Configuration.POLARITY_SELECTION_STRATEGIES.iterkeys(),
                ArgoSAT_Configuration.RESTART_SCHEDULING_STRATEGIES.iterkeys(),
                )

        for names in s:
            yield ArgoSAT_Configuration.from_name(*names)

    @property
    def argv(self):
        arguments = [] 

        if self.variable_selection:
            arguments.append("--variable_selection_strategy")
            arguments.extend(self.variable_selection)

        if self.polarity_selection:
            arguments.append("--literal_polarity_selection_strategy")
            arguments.extend(self.polarity_selection)

        if self.restart_scheduling:
            arguments.append("--restart_strategy")
            arguments.extend(self.restart_scheduling)

        return tuple(arguments)

class SAT_SolverRun(SQL_Base):
    """
    Information about one run of a solver on a SAT task.
    """

    __tablename__ = "sat_solver_runs"

    uuid               = Column(SQL_UUID, primary_key = True, default = uuid4)
    task_uuid          = Column(SQL_UUID, ForeignKey("sat_tasks.uuid"), nullable = False)
    solver_name        = Column(String, ForeignKey("sat_solvers.name"), nullable = False)
    configuration_uuid = Column(SQL_UUID, ForeignKey("sat_solver_configurations.uuid"), nullable = False)
    outcome            = Column(Boolean)
    started            = Column(UTC_DateTime)
    elapsed            = Column(Interval)
    censored           = Column(Boolean)
    fqdn               = Column(String)
    seed               = Column(Integer)

    task          = relation(SAT_Task)
    solver        = relation(SAT_SolverDescription)
    configuration = relation(SAT_SolverConfiguration)

    @staticmethod
    def starting_now(*args, **kwargs):
        """
        Return a partially-initialized run starting now.
        """

        assert "started" not in kwargs
        assert "fqdn" not in kwargs

        run = SAT_SolverRun(*args, **kwargs)

        run.started = utc_now()
        run.fqdn    = socket.getfqdn()

        return run

