"""
utexas/quest/acridid/plot.py

Generate plots.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import logging

from uuid import UUID
from datetime import timedelta
from itertools import (
    izip,
    product,
    )
from sqlalchemy import (
    and_,
    Column,
    Binary,
    String,
    Integer,
    Boolean,
    Interval,
    ForeignKey,
    )
from sqlalchemy.orm import relation
from sqlalchemy.orm.exc import NoResultFound
from cargo.io import (
    hash_file,
    files_under,
    )
from cargo.ai.sat.solvers import ArgoSAT_Solver
from cargo.log import get_logger
from cargo.sql.alchemy import (
    SQL_Base,
    SQL_UUID,
    SQL_Session,
    UTC_DateTime,
    )
from cargo.flags import (
    Flag,
    FlagSet,
    with_flags_parsed,
    )
from cargo.errors import print_ignored_error
from cargo.temporal import utc_now
from utexas.quest.acridid.core import (
    SAT_Task,
    SAT_SolverRun,
    SAT_SolverDescription,
    ArgoSAT_Configuration,
    )

log = get_logger(__name__, level = None)

import math
import pprint
import pylab
from collections import defaultdict

@with_flags_parsed()
def plot_main(positional):
    """
    Application body.
    """

    session             = SQL_Session()
    configuration       = list(yield_argosat_configurations(session))[0]
    argosat_description = get_argosat_solver_description(session)

    query = \
        session.query(SAT_SolverRun).filter(
            and_(
                SAT_SolverRun.solver == argosat_description,
                SAT_SolverRun.configuration == configuration,
                )
            )

    times = defaultdict(list)

    for run in query:
        if run.task.path.startswith("parity/"):
            log_time = math.log(1.0 + run.elapsed.seconds + run.elapsed.microseconds / 1e6)

            times[run.task.path].append(log_time)

    pprint.pprint(dict(times))

if __name__ == '__main__':
#     main()
    plot_main()

