"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

machine_speed = 1.0
minimum_fake_run_budget = 1800.0 # XXX
proc_poll_period = 4.0

try:
    from borg_site_defaults import *
except ImportError:
    pass

