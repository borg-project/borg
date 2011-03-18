"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from __future__ import absolute_import

named_domains = {}

def named_domain(name):
    def decorator(domain_class):
        named_domains[name] = domain_class

        return domain_class

    return decorator

def get_domain(name):
    return named_domains[name]()

from . import defaults
from . import domains
from . import portfolios
from . import dimacs
from . import bilevel
from . import models
from . import opb
from . import expenses

from borg.expenses import *
from borg.domains.features import get_features_for

