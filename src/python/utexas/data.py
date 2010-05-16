"""
utexas/data.py

Storage and retrieval of research data.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from uuid                       import (
    UUID,
    uuid4,
    )
from sqlalchemy                 import (
    Enum,
    Text,
    Table,
    Float,
    Column,
    String,
    Integer,
    Boolean,
    ForeignKey,
    LargeBinary,
    CheckConstraint,
    )
from sqlalchemy.orm             import (
    relationship,
    )
from sqlalchemy.ext.declarative import declarative_base
from cargo.log                  import get_logger
from cargo.sql.alchemy          import (
    SQL_UUID,
    SQL_List,
    SQL_Engines,
    UTC_DateTime,
    SQL_TimeDelta,
    )
from cargo.flags                import (
    Flag,
    Flags,
    )

log          = get_logger(__name__)
DatumBase    = declarative_base()
module_flags = \
    Flags(
        "Research Data Storage",
        Flag(
            "--research-database",
            default = "sqlite:///:memory:",
            metavar = "DATABASE",
            help    = "use research DATABASE by default [%default]",
            ),
        )

def research_connect(engines = SQL_Engines.default, flags = module_flags.given):
    """
    Connect to research data storage.
    """

    flags = module_flags.merged(flags)

    return engines.get(flags.research_database)

class CPU_LimitedRunRow(DatumBase):
    """
    Information about a CPU-limited run of some program.
    """

    __tablename__ = "cpu_limited_runs"

    uuid          = Column(SQL_UUID, primary_key = True, default = uuid4)
    started       = Column(UTC_DateTime)
    usage_elapsed = Column(SQL_TimeDelta)
    proc_elapsed  = Column(SQL_TimeDelta)
    cutoff        = Column(SQL_TimeDelta)
    fqdn          = Column(String)
    stdout        = Column(LargeBinary)
    stderr        = Column(LargeBinary)
    exit_status   = Column(Integer)
    exit_signal   = Column(Integer)

    @staticmethod
    def from_run(run, **kwargs):
        """
        Create a row from a run instance.
        """

        row = \
            CPU_LimitedRunRow(
                started       = run.started,
                usage_elapsed = run.usage_elapsed,
                proc_elapsed  = run.proc_elapsed,
                cutoff        = run.limit,
                exit_status   = run.exit_status,
                exit_signal   = run.exit_signal,
                **kwargs
                )

        row.set_stdout("".join(c for (_, c) in run.out_chunks))
        row.set_stderr("".join(c for (_, c) in run.err_chunks))

        return row

    def get_stdout(self):
        """
        Get stdout text, uncompressed.
        """

        from cargo.io import unxzed

        return unxzed(self.stdout)

    def set_stdout(self, stdout):
        """
        Set stdout text, compressing.
        """

        from cargo.io import xzed

        self.stdout = xzed(stdout)

    def get_stderr(self):
        """
        Get stderr text, uncompressed.
        """

        from cargo.io import unxzed

        return unxzed(self.stderr)

    def set_stderr(self, stderr):
        """
        Set stderr text, compressing.
        """

        from cargo.io import xzed

        self.stderr = xzed(stderr)

class SAT_SolverRow(DatumBase):
    """
    Some solver for some domain.
    """

    __tablename__ = "sat_solvers"

    name = Column(String, primary_key = True)
    type = Column(String)

class SAT_PreprocessorRow(DatumBase):
    """
    Some preprocessor.
    """

    __tablename__ = "sat_preprocessors"

    name = Column(String, primary_key = True)

class SAT_TrialRow(DatumBase):
    """
    A set of [sets of [...]] attempts.
    """

    RECYCLABLE_UUID = UUID("777d15f0-b1cd-4c89-9bf9-814d0974c748")

    __tablename__ = "sat_trials"

    uuid        = Column(SQL_UUID, primary_key = True, default = uuid4)
    parent_uuid = Column(SQL_UUID, ForeignKey("sat_trials.uuid"))
    label       = Column(String)

    @staticmethod
    def get_recyclable(session):
        """
        Retrieve the core "recyclable" trial.
        """

        return                                                         \
            session                                                    \
            .query(SAT_TrialRow)                                       \
            .filter(SAT_TrialRow.uuid == SAT_TrialRow.RECYCLABLE_UUID) \
            .one()

class TaskRow(DatumBase):
    """
    One task.
    """

    __tablename__ = "tasks"
    task_type     = \
        Enum(
            "cnf_sat",
            "preprocessed",
            name = "task_type",
            )

    uuid = Column(SQL_UUID, primary_key = True, default = uuid4)
    type = Column(task_type)

    __mapper_args__ = {"polymorphic_on": type}

class SAT_TaskRow(TaskRow):
    """
    One satisfiability task in DIMACS CNF format.
    """

    TASK_NAMESPACE = UUID("8e67a81a-717c-4206-8831-6007bc8f111f")

    __tablename__   = "sat_tasks"
    __mapper_args__ = {"polymorphic_identity" : "cnf_sat"}

    uuid = Column(SQL_UUID, ForeignKey(TaskRow.uuid), primary_key = True)
    hash = Column(LargeBinary(length = 64))

class PreprocessedTaskRow(TaskRow):
    """
    One task after preprocessing.
    """

    __tablename__ = "preprocessed_tasks"

    uuid              = Column(SQL_UUID, ForeignKey(TaskRow.uuid), primary_key = True)
    preprocessor_name = Column(String, ForeignKey("sat_preprocessors.name"), nullable = False)
    seed              = Column(Integer)
    input_task_uuid   = Column(SQL_UUID, ForeignKey("tasks.uuid"), nullable = False)

    __mapper_args__ = {
        "polymorphic_identity" : "preprocessed",
        "inherit_condition"    : uuid == TaskRow.uuid,
        }

    input_task = \
        relationship(
            TaskRow,
            primaryjoin = (input_task_uuid == TaskRow.uuid),
            )

class TaskNameRow(DatumBase):
    """
    Place a task in the context of a collection.
    """

    __tablename__ = "task_names"

    uuid       = Column(SQL_UUID, primary_key = True, default = uuid4)
    task_uuid  = Column(SQL_UUID, ForeignKey("tasks.uuid"), nullable = False)
    name       = Column(String)
    collection = Column(String)

    task = relationship(TaskRow)

class SAT_AnswerRow(DatumBase):
    """
    Answer to a SAT instance.
    """

    __tablename__ = "sat_answers"

    uuid           = Column(SQL_UUID, primary_key = True, default = uuid4)
    satisfiable    = Column(Boolean)
    certificate_xz = Column(LargeBinary)

    def __init__(self, satisfiable = None, certificate = None, certificate_xz = None):
        """
        Initialize.
        """

        # argument sanity
        if certificate is not None and certificate_xz is not None:
            raise ValueError("cannot specify both certificate and certificate blob")

        # members
        self.satisfiable = satisfiable

        if certificate is not None:
            self.set_certificate(certificate)
        elif certificate_xz is not None:
            self.certificate_xz = certificate_xz

    def get_certificate(self):
        """
        Get the certificate array, uncompressed.
        """

        if self.certificate_xz is None:
            return None
        else:
            return SAT_AnswerRow.unpack_certificate(self.certificate_xz)

    def set_certificate(self, certificate):
        """
        Set (and compress) the certificate array.
        """

        import json

        from cargo.io import xzed

        self.certificate_xz = xzed(json.dumps(certificate))

    @staticmethod
    def unpack_certificate(blob):
        """
        Uncompress and interpret a certificate array.
        """

        import json

        from cargo.io import unxzed

        return json.loads(unxzed(blob))

preprocessor_runs_trials_table = \
    Table(
        "preprocessor_runs_trials",
        DatumBase.metadata,
        Column("preprocessor_run_uuid", SQL_UUID, ForeignKey("preprocessor_runs.uuid")),
        Column("trial_uuid", SQL_UUID, ForeignKey("sat_trials.uuid")),
        )

class PreprocessorRunRow(DatumBase):
    """
    Execution of a preprocessor on a task.
    """

    __tablename__ = "preprocessor_runs"

    uuid              = Column(SQL_UUID, primary_key = True, default = uuid4)
    preprocessor_name = Column(String, ForeignKey("sat_preprocessors.name"), nullable = False)
    input_task_uuid   = Column(SQL_UUID, ForeignKey("tasks.uuid"), nullable = False)
    output_task_uuid  = Column(SQL_UUID, ForeignKey("tasks.uuid"), nullable = False)
    run_uuid          = Column(SQL_UUID, ForeignKey("cpu_limited_runs.uuid"), nullable = False)
    answer_uuid       = Column(SQL_UUID, ForeignKey("sat_answers.uuid"))
    seed              = Column(Integer)
    budget            = Column(SQL_TimeDelta)
    cost              = Column(SQL_TimeDelta)

    input_task  = \
        relationship(
            TaskRow,
            primaryjoin = (input_task_uuid == TaskRow.uuid),
            )
    output_task = \
        relationship(
            TaskRow,
            primaryjoin  = (output_task_uuid == TaskRow.uuid),
            )
    run         = relationship(CPU_LimitedRunRow)
    answer      = relationship(SAT_AnswerRow)
    trials      = \
        relationship(
            SAT_TrialRow,
            secondary = preprocessor_runs_trials_table,
            backref   = "preprocessor_runs",
            )

sat_attempts_trials_table = \
    Table(
        "sat_attempts_trials",
        DatumBase.metadata,
        Column("attempt_uuid", SQL_UUID, ForeignKey("sat_attempts.uuid")),
        Column("trial_uuid", SQL_UUID, ForeignKey("sat_trials.uuid")),
        )

class SAT_AttemptRow(DatumBase):
    """
    An attempt to solve a task.
    """

    __tablename__ = "sat_attempts"
    attempt_type  =\
        Enum(
            "run",
            "portfolio",
            name = "sat_attempt_type",
            )

    uuid        = Column(SQL_UUID, primary_key = True, default = uuid4)
    type        = Column(attempt_type)
    task_uuid   = Column(SQL_UUID, ForeignKey("tasks.uuid"))
    answer_uuid = Column(SQL_UUID, ForeignKey("sat_answers.uuid"))
    budget      = Column(SQL_TimeDelta)
    cost        = Column(SQL_TimeDelta)

    __mapper_args__ = {"polymorphic_on": type}

    task   = relationship(TaskRow)
    answer = relationship(SAT_AnswerRow)
    trials = \
        relationship(
            SAT_TrialRow,
            secondary = sat_attempts_trials_table,
            backref   = "sat_attempts",
            )

class SAT_RunAttemptRow(SAT_AttemptRow):
    """
    An attempt to solve a task with a concrete solver.
    """

    __tablename__   = "sat_run_attempts"
    __mapper_args__ = {"polymorphic_identity": "run"}

    uuid        = Column(SQL_UUID, ForeignKey("sat_attempts.uuid"), primary_key = True)
    run_uuid    = Column(SQL_UUID, ForeignKey("cpu_limited_runs.uuid"), nullable = False)
    solver_name = Column(String, ForeignKey("sat_solvers.name"), nullable = False)
    seed        = Column(Integer)

    run    = relationship(CPU_LimitedRunRow)
    solver = relationship(SAT_SolverRow)

class SAT_PortfolioAttemptRow(SAT_AttemptRow):
    """
    An attempt to solve a task with a portfolio.
    """

    __tablename__   = "sat_portfolio_attempts"
    __mapper_args__ = {"polymorphic_identity": "portfolio"}

    uuid = Column(SQL_UUID, ForeignKey("sat_attempts.uuid"), primary_key = True)

