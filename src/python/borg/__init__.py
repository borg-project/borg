"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

from __future__ import absolute_import

import plac
import os.path
import imp
import uuid
import condor
import borg.log

logger = borg.log.get_logger(__name__, default_level = "INFO")
named_domains = {}

def do(*args, **kwargs):
    return condor.do(*args, **kwargs)

def named_domain(domain_class):
    """Decorates and automatically registers a domain class."""

    named_domains[domain_class.name] = domain_class()

    return domain_class

def get_domain(name):
    """Look up and instantiate a domain."""

    return named_domains[name]

def make_solvers(class_, suite_path, commands):
    root = os.path.abspath(os.path.dirname(suite_path))

    return dict((k, class_(root, v)) for (k, v) in commands.items())

def make_solvers_full(class_, suite_path, arg_lists):
    root = os.path.abspath(os.path.dirname(suite_path))

    return dict((k, class_(root, *v)) for (k, v) in arg_lists.items())

def load_solvers(path):
    """Load a suite of solvers."""

    logger.info("loading solver suite from %s", path)

    return imp.load_source("borg.suite_{0}".format(uuid.uuid4().hex), path)

class Suite(object):
    """Suite of subsolvers."""

    def __init__(self, domain = None, solvers = None):
        """Initialize."""

        self.domain = domain
        self.solvers = {} if solvers is None else solvers

    def integrate(self, other):
        """Integrate another suite into this one."""

        if self.domain is None:
            self.domain = other.domain
        elif self.domain is not other.domain:
            raise ArgumentError("solver suite domains do not match")

        self.solvers.update(other.solvers)

    @staticmethod
    def integrated(*suites):
        """Merge multiple subsolver suites."""

        merged = Suite()

        for suite in suites:
            merged.integrate(suite)

        return merged

    @staticmethod
    def load_integrated(*paths):
        """Load and merge multiple subsolver suites."""

        return Suite.integrated(*map(load_solvers, paths))

def script(main):
    """Call a script main function."""

    borg.enable_default_logging()

    plac.call(main)

annotations = plac.annotations

from . import defaults
from . import util

from borg.log import *

from . import models
from . import planners
from . import expenses
from . import solver_io
from . import storage
from . import fake
from . import unix
from . import bregman
from . import regression
from . import portfolios
from . import domains
from . import experiments
from . import log

from borg.expenses import *
from borg.storage import (
    RunData,
    TrainingData,
    )

