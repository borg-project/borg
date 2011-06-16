"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from __future__ import absolute_import

named_domains = {}

def named_domain(domain_class):
    named_domains[domain_class.name] = domain_class

    return domain_class

def get_domain(name):
    return named_domains[name]()

from . import defaults
from . import portfolios
from . import bilevel
from . import models
from . import expenses
from . import solver_io
from . import storage
from . import domains

from borg.expenses import *

