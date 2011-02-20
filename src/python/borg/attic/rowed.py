"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import abstractmethod
from cargo.sugar import ABC

class NoRowError(Exception):
    """
    An object did not have an associated ORM row.
    """

class AbstractRowed(ABC):
    """
    Abstract base for a class whose objects (may) have associated ORM rows.
    """

    @abstractmethod
    def get_row(self, session):
        """
        Get the ORM row associated with this object, if any.
        """

    @abstractmethod
    def set_row(self, row = None):
        """
        Set the ORM row associated with this object, if any.
        """

class Rowed(AbstractRowed):
    """
    Typical implementation of a class whose objects (may) have associated ORM rows.
    """

    def __init__(self, row = None):
        """
        Initialize.
        """

        self._orm_row = row

    def get_row(self, session, **kwargs):
        """
        Get the ORM row associated with this object, if any.
        """

        if self._orm_row is None:
            row = self.get_new_row(session, **kwargs)

            if row is None:
                raise NoRowError("object has no associated ORM instance")
            else:
                self._orm_row = row

                return row
        else:
            return session.merge(self._orm_row)

    def set_row(self, row = None):
        """
        Set the ORM row associated with this object.
        """

        self._orm_row = row

    def get_new_row(self, session, **kwargs):
        """
        Create or obtain an ORM row for this object.
        """

        return None

