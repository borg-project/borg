"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os

machine_speed = 1.0
proc_poll_period = 1.0
root_log_level = os.environ.get("BORG_LOG_ROOT_LEVEL", "NOTSET")

try:
    from borg_site_defaults import *
except ImportError:
    pass

