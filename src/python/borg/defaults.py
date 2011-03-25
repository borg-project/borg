"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

#solvers_root = "/scratch/cluster/bsilvert/sat-competition-2011/solvers"
#solvers_root = "/scratch/cluster/bsilvert/pb-competition-2011/solvers" # XXX
solvers_root = "/scratch/cluster/bsilvert/max-sat-competition-2011/solvers" # XXX
machine_speed = 1.0
minimum_fake_run_budget = 1800.0 # XXX
proc_poll_period = 1.0

try:
    from borg_site_defaults import *
except ImportError:
    pass

