"""
utexas/quest/acridid/core.py

Common acridid code.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.quest.acridid.core import main

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
    Column,
    Binary,
    String,
    Integer,
    Boolean,
    ForeignKey,
    )
from sqlalchemy.orm import (
    relation,
    sessionmaker,
    )
from sqlalchemy.orm.exc import NoResultFound
from cargo.io import (
    hash_file,
    files_under,
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

log            = get_logger(__name__, level = None)
AcrididBase    = declarative_base()
AcrididSession = sessionmaker()
module_flags   = \
    Flags(
        "Acridid Core Configuration",
        Flag(
            "--acridid-database",
            default = "sqlite:///:memory:",
            metavar = "DATABASE",
            help    = "use DATABASE by default [%default]",
            ),
        )

class SAT_Task(AcrididBase):
    """
    One satisfiability task in DIMACS CNF format.
    """

    __tablename__ = "sat_tasks"

    uuid = Column(SQL_UUID, primary_key = True, default = uuid4)
    hash = Column(Binary(length = 64))
    path = Column(String)

    TASK_NAMESPACE = UUID("f1f6fe7bde244873ba4f7a60d51b8d42")

    @staticmethod
    def from_file(root, path):
        """
        Describe an existing file.
        """

        # hash and retrieve the task
        (_, hash)     = hash_file(path, "sha512")
        relative_path = os.path.relpath(path, root)

        return \
            SAT_Task(
                uuid = \
                    uuid5(
                        SAT_Task.TASK_NAMESPACE,
                        relative_path + hash,
                        ),
                path = relative_path,
                hash = hash,
                )

class SAT_SolverDescription(AcrididBase):
    """
    Information about a SAT solver.
    """

    __tablename__ = "sat_solvers"

    name = Column(String, primary_key = True)
    type = Column(String)

    __mapper_args__ = {"polymorphic_on": type}

class SAT_2007_SolverDescription(SAT_SolverDescription):
    """
    Configuration of a solver from the 2007 SAT competition.
    """

    __tablename__   = "sat_2007_solvers"
    __mapper_args__ = {"polymorphic_identity": "sat2007"}

    name          = Column(String, ForeignKey(SAT_SolverDescription.name), primary_key = True)
    relative_path = Column(String)
    seeded        = Column(Boolean)

class SAT_ConfigurationSet(AcrididBase):
    """
    Related set of job records.
    """

    __tablename__ = "configuration_sets"

    uuid = Column(SQL_UUID, primary_key = True, default = uuid4)
    name = Column(String)

class SAT_SolverConfiguration(AcrididBase):
    """
    Configuration of a SAT solver.
    """

    __tablename__ = "sat_solver_configurations"

    uuid        = Column(SQL_UUID, primary_key = True, default = uuid4)
    name        = Column(String)
    solver_type = Column(String)
    set_uuid    = Column(SQL_UUID, ForeignKey("configuration_sets.uuid"))

    set = relation(SAT_ConfigurationSet)

    __mapper_args__ = {"polymorphic_on": solver_type}

class ArgoSAT_Configuration(SAT_SolverConfiguration):
    """
    Configuration of ArgoSAT.
    """

    __tablename__   = "argosat_configurations"
    __mapper_args__ = {"polymorphic_identity": "argosat"}

    uuid              = Column(
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
        """
        Create a configuration from names using a deterministic UUID.
        """

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
                uuid               = \
                    uuid5(
                        ArgoSAT_Configuration.NAMED_CONFIGURATION_NAMESPACE,
                        name,
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

class SATensteinConfiguration(SAT_SolverConfiguration):
    """
    Configuration of SATenstein.
    """

    __tablename__   = "satenstein_configurations"
    __mapper_args__ = {"polymorphic_identity": "satenstein"}

    uuid       = Column(
        SQL_UUID,
        ForeignKey("sat_solver_configurations.uuid"),
        default     = uuid4,
        primary_key = True,
        )
    parameters = Column(SQL_JSON)

    NAMED_CONFIGURATION_NAMESPACE = UUID("4cdba386aa224cd5b575cf556672e0a3")

    @staticmethod
    def from_parameters(parameters, set):
        """
        Generate a configuration from a parameter dictionary.
        """

        return \
            SATensteinConfiguration(
                uuid       = \
                    uuid5(
                        SATensteinConfiguration.NAMED_CONFIGURATION_NAMESPACE,
                        str(sorted(parameters.iteritems())) + "#" + str(set.uuid),
                        ),
                parameters = parameters,
                set        = set,
                )

class SAT_SolverRun(AcrididBase):
    """
    Information about one run of a solver on a SAT task.
    """

    __tablename__ = "sat_solver_runs"

    uuid               = Column(SQL_UUID, primary_key = True, default = uuid4)
    task_uuid          = Column(SQL_UUID, ForeignKey("sat_tasks.uuid"), nullable = False)
    solver_name        = Column(String, ForeignKey("sat_solvers.name"), nullable = False)
    configuration_uuid = Column(SQL_UUID, ForeignKey("sat_solver_configurations.uuid"))
    outcome            = Column(Boolean)
    started            = Column(UTC_DateTime)
    elapsed            = Column(SQL_TimeDelta)
    cutoff             = Column(SQL_TimeDelta)
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
        run.fqdn    = getfqdn()

        return run

def acridid_connect(engines = SQL_Engines.default, flags = module_flags.given):
    """
    Connect to acridid storage.
    """

    flags  = module_flags.merged(flags)
    engine = engines.get(flags.acridid_database)

    AcrididBase.metadata.create_all(engine)

    return engine

@with_flags_parsed()
def main(positional):
    """
    Create core database metadata.
    """

    acridid_connect()

