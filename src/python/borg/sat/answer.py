"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

class SAT_Answer(object):
    """
    An answer to a CNF SAT instance.
    """

    def __init__(self, satisfiable, certificate = None):
        """
        Initialize.
        """

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

