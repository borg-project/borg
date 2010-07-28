"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log            import get_logger
from borg.portfolio.world import (
    Action,
    Outcome,
    AbstractTrainer,
    )

log = get_logger(__name__)

class DecisionTrainer(AbstractTrainer):
    """
    Grant a decision portfolio access to training data.
    """

    def __init__(self, task_uuids, Session, extrapolation = 1):
        """
        Initialize.
        """

        self._task_uuids    = task_uuids
        self._Session       = Session
        self._extrapolation = extrapolation

    def build_actions(self, request):
        """
        Build a list of actions from a configuration request.
        """

        # build the solvers and cutoffs
        from cargo.temporal import TimeDelta
        from borg.solvers   import LookupSolver

        solvers = [LookupSolver(s) for s in request["solvers"]]
        budgets = [TimeDelta(seconds = s) for s in request["budgets"]]

        # build the actions
        from itertools import product

        return [DecisionWorldAction(*a) for a in product(solvers, budgets)]

    def get_data(self, action):
        """
        Provide a tasks-by-outcomes array to the trainee.

        Outcomes order matches that of the action.
        """

        with self._Session() as session:
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
                            count(case([(RAR.cost <= action.cost, RAR.answer_uuid)])),
                            count(),
                            ],
                        and_(
                            RAR.solver           == solver_row,
                            RAR.budget           >= action.cost,
                            RAR.__table__.c.uuid == AR.uuid,
                            RAR.task_uuid.in_(self._task_uuids),
                            RAR.trials.contains(recyclable_trial_row),
                            ),
                        group_by = RAR.task_uuid,
                        ),
                    )                                                                  \
                    .fetchall()

            log.detail("got %i rows for action %s", len(rows), action.description)

            # packaged as an array
            import numpy

            e = self._extrapolation

            return numpy.array([[s * e, (a - s) * e] for (s, a) in rows], numpy.uint)

class DecisionWorldAction(Action):
    """
    An action in the world.
    """

    def __init__(self, solver, budget):
        """
        Initialize.
        """

        Action.__init__(self, budget.as_s)

        self._solver = solver
        self._budget = budget

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

        return DecisionWorldOutcome.BY_INDEX

    @property
    def solver(self):
        """
        The solver associated with this action.
        """

        return self._solver

class DecisionWorldOutcome(Outcome):
    """
    An outcome of an action in the world.
    """

    def __init__(self, n, utility):
        """
        Initialize.
        """

        Outcome.__init__(self, utility)

    @staticmethod
    def from_result(attempt):
        """
        Return an outcome from a solver attempt.
        """

        if attempt.answer is None:
            return DecisionWorldOutcome.UNSOLVED
        else:
            return DecisionWorldOutcome.from_bool(attempt.answer.satisfiable)

    @staticmethod
    def from_bool(bool):
        """
        Return an outcome from True, False, or None.
        """

        return DecisionWorldOutcome.BY_VALUE[bool]

# outcome constants
DecisionWorldOutcome.SOLVED   = DecisionWorldOutcome(0, 1.0)
DecisionWorldOutcome.UNSOLVED = DecisionWorldOutcome(1, 0.0)
DecisionWorldOutcome.BY_VALUE = {
    True:  DecisionWorldOutcome.SOLVED,
    False: DecisionWorldOutcome.SOLVED,
    None:  DecisionWorldOutcome.UNSOLVED,
    }
DecisionWorldOutcome.BY_INDEX = [
    DecisionWorldOutcome.SOLVED,
    DecisionWorldOutcome.UNSOLVED,
    ]

