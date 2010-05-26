"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from borg.rowed import Rowed

class Decision(Rowed):
    """
    An answer to an instance of a decision problem.
    """

    def __init__(self, satisfiable, certificate = None, row = None):
        """
        Initialize.
        """

        Rowed.__init__(self, row)

        if satisfiable is None:
            raise ValueError("answer must provide satisfiability")

        self.satisfiable = satisfiable
        self.certificate = certificate

    def __eq__(self, other):
        """
        Is this answer equal to another?
        """

        return                                        \
            type(self) is type(other)                 \
            and self.satisfiable == other.satisfiable \
            and self.certificate == other.certificate

    def __ne__(self, other):
        """
        Is this answer not equal to another?
        """

        return not self == other

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        from borg.data import DecisionRow

        decision_row = DecisionRow(self.satisfiable, self.certificate)

        session.add(decision_row)

        return decision_row

    @staticmethod
    def to_row(decision, session):
        """
        Convert a (possibly-None) decision to a (possibly-None) row.
        """

        if decision is None:
            return None
        else:
            return decision.get_row(session)

