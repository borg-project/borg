"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc                  import (
    abstractmethod,
    abstractproperty,
    )
from contextlib           import contextmanager
from sqlalchemy           import (
    Table,
    Column,
    MetaData,
    )
from cargo.log            import get_logger
from cargo.sugar          import ABC
from borg.portfolio._base import (
    Action,
    Outcome,
    )

logger = get_logger(__name__)

class SolverActionSet(object):
    """
    Retrieve training data for multiple solver actions.
    """

    def __init__(self, actions = []):
        """
        Initialize.
        """

        self._actions = actions

    def add_action(self, action):
        """
        Add an action.
        """

        self._actions.append(action)

    def get_training(self, session, task_table, tasks):
        """
        Return outcomes for each action and task.

        Typically invoked through a trainer.
        """

        # retrieve the outcome summaries
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

        actions_table = self._create_actions_table(session)
        rows          =                                                           \
            session.execute(
                select(
                    [
                        actions_table.c.id,
                        RAR.task_uuid,
                        count(case([(RAR.cost <= actions_table.c.duration, RAR.answer_uuid)])),
                        count(),
                        ],
                    and_(
                        RAR.solver_name      == actions_table.c.solver_name,
                        RAR.budget           >= actions_table.c.duration   ,
                        RAR.task_uuid        == task_table.c.uuid          ,
                        RAR.__table__.c.uuid == AR.uuid                    ,
                        ),
                    group_by = [actions_table.c.id, RAR.task_uuid],
                    ),
                )

        # package the outcomes
        actions = self._actions
        pairs   = [((actions[i], u), (k, n)) for (i, u, k, n) in rows]

        actions_table.drop()

        return dict(pairs)

    def _create_actions_table(self, session):
        """
        Build a temporary actions table.
        """

        from sqlalchemy import (
            String,
            Integer,
            Interval,
            )

        metadata      = MetaData(bind = session.connection())
        actions_table = \
            Table(
                "trainer_solver_actions_temporary",
                metadata,
                Column("id"         , Integer  , primary_key = True),
                Column("solver_name", String                       ),
                Column("duration"   , Interval                     ),
                prefixes = ["temporary"],
                )

        actions_table.create(checkfirst = False)

        session.execute(
            actions_table.insert(),
            [
                {
                    "id"          : i             ,
                    "solver_name" : a._solver.name,
                    "duration"    : a.budget      ,
                    }
                for (i, a) in enumerate(self._actions)
                ],
            )

        return actions_table

class FeatureActionSet(object):
    """
    Retrieve training data for multiple feature actions.
    """

    def __init__(self, actions = []):
        """
        Initialize.
        """

        self._actions = actions

    def add_action(self, action):
        """
        Add an action.
        """

        self._actions.append(action)

    def get_training(self, session, task_table, tasks):
        """
        Return outcomes for each action and task.

        Typically invoked through a trainer.
        """

        analyzer_data = {}
        training_data = {}

        for action in self._actions:
            feature_data = analyzer_data.get(action.analyzer)

            if feature_data is None:
                feature_data                   = \
                analyzer_data[action.analyzer] = \
                    action.analyzer.get_training(session, task_table)

            for task in tasks:
                training_data[(action, task.uuid)] = feature_data[(action.feature, task.uuid)]

        return training_data

class SolverAction(Action):
    """
    An action that executes a solver under some budget constraint.
    """

    solved_outcome   = Outcome(1.0)
    unsolved_outcome = Outcome(0.0)
    outcomes_        = [solved_outcome, unsolved_outcome]
    ActionSet        = SolverActionSet

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
    An action that acquires a static binary feature of a problem instance.
    """

    ActionSet = FeatureActionSet

    def __init__(self, analyzer, feature):
        """
        Initialize.
        """

        Action.__init__(self, 0.0)

        self._analyzer = analyzer
        self._feature  = feature
        self._outcomes = [Outcome(0.0) for i in xrange(2)]

    def __reduce__(self):
        """
        Reduce this instance for pickling.
        """

        return (BinaryFeatureAction, (self._analyzer, self._feature))

    def get_training(self, session, task_table):
        """
        Return a tasks-by-outcomes array.
        """

        return self._analyzer.get_training(session, self._feature, task_table)

    def take(self, features):
        """
        Return an outcome from the relevant feature value.
        """

        return self._outcomes[features[self._feature.name]]

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

        return self._outcomes

    @property
    def analyzer(self):
        """
        The analyzer associated with this action.
        """

        return self._analyzer

    @property
    def feature(self):
        """
        The feature associated with this action.
        """

        return self._feature

class DiscreteFeatureAction(Action):
    """
    An action that acquires a static feature of a problem instance.
    """

    def __init__(self, analyzer, feature):
        """
        Initialize.
        """

        Action.__init__(self, 0.0)

        self._analyzer = analyzer
        self._feature  = feature
        self._outcomes = [Outcome(0.0) for i in xrange(feature.dimensionality)]

    def __reduce__(self):
        """
        Reduce this instance for pickling.
        """

        return (DiscreteFeatureAction, (self._analyzer, self._feature))

    def get_training(self, session, task_table):
        """
        Return a tasks-by-outcomes array.
        """

        return self._analyzer.get_training(session, self._feature, task_table)

    def take(self, features):
        """
        Return an outcome from the relevant feature value.
        """

        return self._outcomes[features[self._feature.name]]

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

        return self._outcomes

class DecisionTrainer(object):
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
        Provide tasks-by-outcomes arrays.
        """

        # group actions by action set
        action_sets = {}

        for action in actions:
            ActionSet  = action.ActionSet
            action_set = action_sets.get(ActionSet)

            if action_set is None:
                action_sets[ActionSet] = action_set = ActionSet()

            action_set.add_action(action)

        # query the database for action data
        with self.context() as (session, task_table):
            outcomes = {}

            for action_set in action_sets.values():
                for_action_set = action_set.get_training(session, task_table, self._tasks)

                outcomes.update(for_action_set)

            return outcomes

    @contextmanager
    def context(self):
        """
        Provide a session instance and training tasks table.
        """

        with self._Session() as session:
            # store the task uuids in a temporary table
            from cargo.sql.alchemy import SQL_UUID

            metadata   = MetaData(bind = session.connection())
            task_table = \
                Table(
                    "trainer_tasks_temporary",
                    metadata,
                    Column("uuid", SQL_UUID, primary_key = True),
                    prefixes = ["temporary"],
                    )

            metadata.create_all(session.connection(), checkfirst = False)

            session.execute(
                task_table.insert(),
                [{"uuid" : t.uuid} for t in self._tasks],
                )

            # the context
            yield (session, task_table)

            # clean up
            task_table.drop()

    @property
    def Session(self):
        """
        Associated session class.
        """

        return self._Session

    @property
    def tasks(self):
        """
        Associated tasks.
        """

        return self._tasks

