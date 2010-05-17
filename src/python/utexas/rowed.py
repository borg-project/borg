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

    def get_row(self, session):
        """
        Get the ORM row associated with this object, if any.
        """

        if self._orm_row is None:
            raise NoRowError()
        else:
            return session.merge(self._orm_row)

    def set_row(self, row = None):
        """
        Set the ORM row associated with this object, if any.
        """

        self._orm_row = row

