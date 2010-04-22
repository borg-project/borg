"""
utexas/sat/tasks.py

Satisfiability task types.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import (
    abstractmethod,
    abstractproperty,
    )
from cargo.log   import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

class SAT_Task(ABC):
    """
    A satisfiability task.
    """

    @abstractproperty
    def name(self):
        """
        An arbitrary printable name for the task.
        """

class SAT_FileTask(SAT_Task):
    """
    A task backed by a .cnf file.
    """

    def __init__(self, path):
        """
        Initialize.
        """

        self.path = path

    @property
    def name(self):
        """
        An arbitrary printable name for the task.
        """

        return self.path

class SAT_MockTask(SAT_Task):
    """
    An abstract task.
    """

class SAT_MockFileTask(SAT_MockTask):
    """
    An abstract file task.
    """

    # FIXME

class SAT_MockPreprocessedTask(SAT_MockTask):
    """
    An abstract preprocessed task.
    """

    # FIXME

