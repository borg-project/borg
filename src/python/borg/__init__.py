"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from tasks     import *
from solvers   import *
from analyzers import *
from portfolio import *

def get_support_path(name):
    """
    Return the absolute path to a support file.
    """

    from os.path import (
        join,
        exists,
        dirname,
        )

    path = join(dirname(__file__), "_files", name)

    if exists(path):
        return path
    else:
        raise RuntimeError("specified support file does not exist")

def export_clean_defaults_path():
    """
    Export a PYTHONPATH
    """

    from os import environ

    environ["PYTHONPATH"] = \
        "%s:%s" % (
            get_support_path("for_tests/reset_defaults"),
            environ["PYTHONPATH"],
            )

