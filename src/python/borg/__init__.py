"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

from __future__ import absolute_import

import imp

named_domains = {}

def named_domain(domain_class):
    """Decorates and automatically registers a domain class."""

    named_domains[domain_class.name] = domain_class

    return domain_class

def get_domain(name):
    """Look up and instantiate a domain."""

    return named_domains[name]()

def load_solvers(path):
    """Load a bundle of solvers."""

    return imp.load_source("borg.solvers_bundle", path)

from . import defaults
from . import portfolios
from . import bilevel
from . import models
from . import expenses
from . import solver_io
from . import storage
from . import domains

from borg.expenses import *

