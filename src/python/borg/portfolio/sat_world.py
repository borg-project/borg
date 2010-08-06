"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log                     import get_logger
from borg.portfolio.world          import (
    Action,
    Outcome,
    )
from borg.portfolio.decision_world import (
    Action,
    Outcome,
    )

log = get_logger(__name__)

class SAT_Trainer(DecisionTrainer):
    """
    Grant a decision portfolio access to training data.
    """

    def __init__(self, task_uuids, Session):
        """
        Initialize.
        """

        DecisionTrainer.__init__(task_uuids, Session)

    @staticmethod
    def build(request):
        """
        Build this trainer from a request.
        """

