"""
utexas/quest/acridid/plot.py

Generate plots.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import math
import pprint
import numpy
import pylab

from collections import defaultdict
from cargo.log import get_logger
from cargo.sql.alchemy import SQL_Session
from cargo.flags import with_flags_parsed
from cargo.sugar import TimeDelta
from utexas.quest.acridid.core import (
    SAT_Task,
    SAT_SolverRun,
    SAT_SolverDescription,
    ArgoSAT_Configuration,
    )

log = get_logger(__name__, level = None)

def plot_variance_histogram(session):
    """
    Plot relevant tasks.
    """

    solver_description   = session.merge(SAT_SolverDescription(name = "argosat"))
    solver_configuration = session.merge(ArgoSAT_Configuration.from_names("r", "r", "n"))
    variances            = {}

    for (seed,) in session.query(SAT_SolverRun.seed).distinct():
        runs_query =                                                    \
            session                                                     \
            .query(SAT_Task, SAT_SolverRun)                             \
            .filter(
                (SAT_SolverRun.seed == seed)
                & (SAT_Task.uuid == SAT_SolverRun.task_uuid)
                & SAT_Task.path.startswith("satlib/dimacs/parity")
                & (SAT_SolverRun.solver == solver_description)
                & (SAT_SolverRun.configuration == solver_configuration)
                )                                                       \
            .order_by(SAT_Task.path)

        if runs_query.count() >= 30:
            variances[seed] = numpy.var([math.log(1.0 + r.elapsed.as_s) for (_, r) in runs_query])

    pylab.hist(variances.values())

def plot_per_seed_lines(session):
    """
    Plot relevant tasks.
    """

    solver_description   = session.merge(SAT_SolverDescription(name = "argosat"))
    solver_configuration = session.merge(ArgoSAT_Configuration.from_names("r", "r", "n"))

    for (seed,) in session.query(SAT_SolverRun.seed).distinct()[:32]:
        runs_query =                                                    \
            session                                                     \
            .query(SAT_Task, SAT_SolverRun)                             \
            .filter(
                (SAT_SolverRun.seed == seed)
                & (SAT_Task.uuid == SAT_SolverRun.task_uuid)
                & SAT_Task.path.startswith("satlib/dimacs/parity")
                & (SAT_SolverRun.solver == solver_description)
                & (SAT_SolverRun.configuration == solver_configuration)
                )                                                       \
            .order_by(SAT_Task.path)

        pylab.plot(
            numpy.arange(runs_query.count()),
            [math.log(1.0 + r.elapsed.as_s) for (_, r) in runs_query],
#             linestyle = " ",
            marker    = "o",
            )

def plot_per_task_elapsed_matrix(session):
    """
    Plot relevant tasks.
    """

    solver_description   = session.merge(SAT_SolverDescription(name = "argosat"))
    solver_configuration = session.merge(ArgoSAT_Configuration.from_names("r", "r", "n"))
    tasks_query          = session.query(SAT_Task).filter(SAT_Task.path.startswith("satlib/dimacs/parity")).order_by(SAT_Task.path)
    nbins                = 8
    elapsed_matrix       = numpy.empty((nbins, tasks_query.count()))

    for (i, task) in enumerate(tasks_query):
        runs_query           = \
            session.query(SAT_SolverRun).filter(
                (SAT_SolverRun.task == task)
                & (SAT_SolverRun.solver == solver_description)
                & (SAT_SolverRun.configuration == solver_configuration)
                )
        (histogram, _)       = \
            numpy.histogram(
                numpy.array([math.log(1.0 + r.elapsed.as_s) for r in runs_query]),
                bins  = nbins,
                range = (0.0, math.log(512.0)),
                )
        elapsed_matrix[:, i] = histogram

    pylab.bone()
    pylab.matshow(elapsed_matrix, fignum = False)

@with_flags_parsed()
def main(positional):
    """
    Application body.
    """

    session = SQL_Session()

    pylab.figure(1)
    plot_per_task_elapsed_matrix(session)
    pylab.figure(2)
    plot_per_seed_lines(session)
    pylab.figure(3)
    plot_variance_histogram(session)

    pylab.show()

if __name__ == '__main__':
    main()

