"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

from __future__ import absolute_import

import os.path
import imp
import uuid
import cargo

logger = cargo.get_logger(__name__, default_level = "INFO")
named_domains = {}

def named_domain(domain_class):
    """Decorates and automatically registers a domain class."""

    named_domains[domain_class.name] = domain_class()

    return domain_class

def get_domain(name):
    """Look up and instantiate a domain."""

    return named_domains[name]

def make_solvers(class_, suite_path, commands):
    root = os.path.dirname(os.path.abspath(suite_path))

    return dict((k, class_(root, v)) for (k, v) in commands.items())

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

from . import defaults
from . import portfolios
from . import models
from . import expenses
from . import solver_io
from . import storage
from . import fake
from . import planners
from . import domains

from borg.expenses import *
from borg.storage import TrainingData

