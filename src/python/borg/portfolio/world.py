"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc                  import (
    abstractmethod,
    abstractproperty,
    )
from cargo.sugar          import ABC
from borg.portfolio._base import (
    Action,
    Outcome,
    )

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

        Action.__init__(self, budget.as_s)

        self._solver = solver
        self._budget = budget

    def get_training(self, session, task_uuids):
        """
        Return a tasks-by-outcomes array.
        """

        from sqlalchemy               import (
            and_,
            case,
            select,
            )
        from sqlalchemy.sql.functions import count
        from borg.data                import (
            TrialRow      as TR,
            AttemptRow    as AR,
            RunAttemptRow as RAR,
            )

        # related rows
        solver_row           = action.solver.get_row(session)
        recyclable_trial_row = TR.get_recyclable(session)

        # existing action outcomes
        rows =                                                                     \
            session.execute(
                select(
                    [
                        RAR.task_uuid,
                        count(case([(RAR.cost <= action.cost, RAR.answer_uuid)])),
                        count(),
                        ],
                    and_(
                        RAR.solver           == solver_row,
                        RAR.budget           >= action.cost,
                        RAR.__table__.c.uuid == AR.uuid,
                        RAR.task_uuid.in_(task_uuids),
                        RAR.trials.contains(recyclable_trial_row),
                        ),
                    group_by = RAR.task_uuid,
                    ),
                )                                                                  \
                .fetchall()

        # FIXME ordering

        log.detail("fetched %i rows for action %s", len(rows), action.description)

        # build the array
        import numpy

        return numpy.array([[s, (a - s)] for (s, a) in rows], numpy.uint)

    def take(self, task, remaining, random, environment):
        """
        Take the action.
        """

        from cargo.temporal import TimeDelta

        calibrated = TimeDelta(seconds = self.cost * environment.time_ratio)
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

    @staticmethod
    def build_actions(request):
        """
        Build a sequence of solver actions.
        """

        # build the solvers and cutoffs
        from cargo.temporal import TimeDelta
        from borg.solvers   import LookupSolver

        solvers = [LookupSolver(s) for s in request["solvers"]]
        budgets = [TimeDelta(seconds = s) for s in request["budgets"]]

        # build the actions
        from itertools import product

        return [SolverAction(*a) for a in product(solvers, budgets)]

class FeatureAction(Action):
    """
    An action that acquires a static feature of a problem instance.
    """

    outcomes_ = [Outcome(0.0), Outcome(0.0)]

    def __init__(self, feature_name):
        """
        Initialize.
        """

        Action.__init__(self, 0.0)

        self._feature_name = feature_name

    def get_training(self, session, task_uuids):
        """
        Return a tasks-by-outcomes array.
        """

        from sqlalchemy import and_
        from borg.data  import (
            TaskRow        as TR,
            TaskFeatureRow as TFR,
            )

        # existing action outcomes
        rows =                                          \
            session.execute(
                select(
                    [TFR.task_uuid, TFR.value],
                    and_(
                        TFR.name == self._feature_name,
                        TFR.task_uuid.in_(task_uuids),
                        ),
                    ),
                )                                       \
                .fetchall()

        log.detail("fetched %i rows for feature %s", len(rows), action.description)

        # build the array
        import numpy

        mapped = dict(rows)
        counts = {
            True  : [0, 1],
            False : [1, 0],
            }

        return numpy.array([counts[mapped[u]] for u in task_uuids], numpy.uint)

    def take(self, features):
        """
        Return an outcome from the relevant feature value.
        """

        if features[self._feature_name]:
            return SolverAction.unsolved_outcome
        else:
            return SolverAction.solved_outcome

    @property
    def description(self):
        """
        A human-readable name for this action.
        """

        return self._feature_name

    @property
    def outcomes(self):
        """
        The possible outcomes of this action.
        """

        return FeatureAction.outcomes_

    @property
    def feature_name(self):
        """
        The name of the associated feature.
        """

        return self._feature_name

class Trainer(ABC):
    """
    Grant a portfolio access to training data.
    """

    @abstractmethod
    def get_data(self, action):
        """
        Provide task-outcomes counts to the trainee.
        """

    @abstractproperty
    def actions(self):
        """
        Return the associated actions.
        """

    @staticmethod
    def build(Session, task_uuids, request):
        """
        Build a trainer as requested.
        """

        builders = {
            "pb"  : PB_Trainer.build,
            "sat" : SAT_Trainer.build,
            }

        return builders[request["type"]](Session, task_uuids, request)

class PB_Trainer(Trainer):
    """
    Grant a decision portfolio access to training data.
    """

    def __init__(self, Session, task_uuids, actions):
        """
        Initialize.
        """

        self._Session    = Session
        self._task_uuids = task_uuids
        self._actions    = actions

    def get_data(self, action):
        """
        Provide a tasks-by-outcomes array to the trainee.
        """

        with self._Session() as session:
            return action.get_training(session, self._task_uuids)

    @property
    def actions(self):
        """
        Return the associated actions.
        """

        return self._actions

    @staticmethod
    def build(Session, task_uuids, request):
        """
        Build this trainer from a request.
        """

        actions = SolverAction.build_actions(request["actions"])

        return PB_Trainer(Session, task_uuids, actions)

class SAT_Trainer(Trainer):
    """
    Grant a decision portfolio access to training data.
    """

    def __init__(self, Session, task_uuids, actions):
        """
        Initialize.
        """

        self._Session    = Session
        self._task_uuids = task_uuids
        self._actions    = actions

    def get_data(self, action):
        """
        Provide a tasks-by-outcomes array to the trainee.
        """

        with self._Session() as session:
            return action.get_training(session, self._task_uuids)

    @property
    def actions(self):
        """
        Return the associated actions.
        """

        return self._actions

    @staticmethod
    def build(Session, task_uuids, request):
        """
        Build this trainer from a request.
        """

        from borg.sat.cnf import feature_names

        # FIXME action construction doesn't belong here...?

        actions  = SolverAction.build_actions(request["actions"])
        actions += [FeatureAction(name) for name in feature_names]

        return SAT_Trainer(Session, task_uuids, actions)

