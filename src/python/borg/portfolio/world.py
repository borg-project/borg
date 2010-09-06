"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc                  import (
    abstractmethod,
    abstractproperty,
    )
from cargo.log            import get_logger
from cargo.sugar          import ABC
from borg.portfolio._base import (
    Action,
    Outcome,
    )

log = get_logger(__name__)

class SolverAction(Action):
    """
    An action that executes a solver under some budget constraint.
    """

    solved_outcome   = Outcome(1.0)
    unsolved_outcome = Outcome(0.0)
    outcomes_        = [solved_outcome, unsolved_outcome]

    def __init__(self, solver, budget):
        """
        Initialize.
        """

        from cargo.temporal import seconds

        Action.__init__(self, seconds(budget))

        self._solver = solver
        self._budget = budget

    def __reduce__(self):
        """
        Reduce this instance for pickling.
        """

        return (SolverAction, (self._solver, self._budget))

    def get_training(self, session, tasks):
        """
        Return a tasks-by-outcomes array.

        Typically invoked through a trainer.
        """

        import numpy

        from sqlalchemy               import (
            and_,
            case,
            select,
            )
        from sqlalchemy.sql.functions import count
        from borg.data                import (
            AttemptRow    as AR,
            RunAttemptRow as RAR,
            )

        task_uuids = [t.get_row(session).uuid for t in tasks]
        rows       =                                                               \
            session.execute(
                select(
                    [
                        count(case([(RAR.cost <= self.cost, RAR.answer_uuid)])),
                        count(),
                        ],
                    and_(
                        RAR.solver           == self._solver.get_row(session),
                        RAR.budget           >= self.cost,
                        RAR.__table__.c.uuid == AR.uuid,
                        RAR.task_uuid.in_(task_uuids),
                        ),
                    group_by = RAR.task_uuid,
                    order_by = RAR.task_uuid,
                    ),
                )                                                                  \
                .fetchall()

        if len(rows) != len(tasks):
            log.warning(
                "fetched only %i rows for action %s on %i tasks",
                 len(rows),
                 len(tasks),
                 self.description,
                 )
        else:
            log.detail("fetched %i rows for action %s", len(rows), self.description)

        order     = sorted(xrange(len(task_uuids)), key = lambda i: task_uuids[i])
        unordered = numpy.array([[s, (a - s)] for (s, a) in rows], numpy.uint)

        return unordered[numpy.array(order)]

    def take(self, task, remaining, random, environment):
        """
        Take the action.
        """

        from datetime import timedelta

        calibrated = timedelta(seconds = self.cost * environment.time_ratio)
        attempt    = self._solver.solve(task, min(remaining, calibrated), random, environment)

        if attempt.answer is None:
            return (attempt, SolverAction.unsolved_outcome)
        else:
            return (attempt, SolverAction.solved_outcome)

    @property
    def description(self):
        """
        A human-readable description of this action.
        """

        return "%s_%ims" % (self.solver.name, int(self.cost * 1000))

    @property
    def budget(self):
        """
        The time-valued cost of taking this action.
        """

        return self._budget

    @property
    def outcomes(self):
        """
        The possible outcomes of this action.
        """

        return SolverAction.outcomes_

    @property
    def solver(self):
        """
        The solver associated with this action.
        """

        return self._solver

class BinaryFeatureAction(Action):
    """
    An action that acquires a static feature of a problem instance.
    """

    outcomes_ = [Outcome(0.0), Outcome(0.0)]

    def __init__(self, analyzer, feature):
        """
        Initialize.
        """

        Action.__init__(self, 0.0)

        self._analyzer = analyzer
        self._feature  = feature

    def __reduce__(self):
        """
        Reduce this instance for pickling.
        """

        return (BinaryFeatureAction, (self._analyzer, self._feature))

    def get_training(self, session, tasks):
        """
        Return a tasks-by-outcomes array.
        """

        return self._analyzer.get_training(session, self._feature, tasks)

    def take(self, features):
        """
        Return an outcome from the relevant feature value.
        """

        if features[self._feature.name]:
            return BinaryFeatureAction.outcomes_[0]
        else:
            return BinaryFeatureAction.outcomes_[1]

    @property
    def description(self):
        """
        A human-readable name for this action.
        """

        return self._feature.name

    @property
    def outcomes(self):
        """
        The possible outcomes of this action.
        """

        return BinaryFeatureAction.outcomes_

class Trainer(ABC):
    """
    Grant a portfolio access to training data.
    """

    @abstractmethod
    def get_data(self, actions):
        """
        Provide task-outcomes counts to the trainee.
        """

class DecisionTrainer(Trainer):
    """
    Grant a decision portfolio access to training data.
    """

    def __init__(self, Session, tasks):
        """
        Initialize.
        """

        self._Session = Session
        self._tasks   = tasks

    def get_data(self, actions):
        """
        Provide a tasks-by-outcomes array to the trainee.
        """

        with self._Session() as session:
            return [a.get_training(session, self._tasks) for a in actions]

