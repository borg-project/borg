# vim: set fileencoding=UTF-8 :
"""
utexas/tools/portfolio/practice.py

Collect training data.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.portfolio.practice import main

    raise SystemExit(main())

import json
import logging
import numpy

from os                         import getenv
from os.path                    import join
from itertools                  import product
from cargo.log                  import get_logger
from cargo.json                 import jsonify
from cargo.flags                import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.temporal             import TimeDelta
from utexas.sat.solvers         import get_named_solvers
from utexas.portfolio.sat_world import (
    SAT_WorldTask,
    SAT_WorldAction,
    )

log = get_logger(__name__, level = logging.NOTSET)

# FIXME use sqlite for run storage instead?

module_flags = \
    Flags(
        "Script Options",
        Flag(
            "-r",
            "--restarts",
            default = 16,
            type    = int,
            metavar = "INT",
            help    = "make INT restarts on each task [%default]",
            ),
        Flag(
            "-t",
            "--tasks",
            default = "tasks.json",
            metavar = "FILE",
            help    = "load task names from FILE [%default]",
            ),
        Flag(
            "-s",
            "--solvers",
            default = "solver_names.json",
            metavar = "FILE",
            help    = "load solver names from FILE [%default]",
            ),
        Flag(
            "-c",
            "--cutoffs",
            default = "cutoffs.json",
            metavar = "FILE",
            help    = "load cutoff times from FILE [%default]",
            ),
        Flag(
            "-o",
            "--output",
            default = "history.json",
            metavar = "FILE",
            help    = "write practice history to FILE [%default]",
            ),
        )

def practice(action, tasks, nrestarts):
    """
    Practice an action on a task.
    """

    log.detail("practicing %s", action)

    indices = dict((o, n) for (n, o) in enumerate(action.outcomes))

    assert len(indices) == len(action.outcomes)

    def yield_counts():
        """
        Yield the counts for each task, in order.
        """

        for task in tasks:
            log.debug("on %s", task.path)

            counts = numpy.zeros(len(indices), numpy.int)

            for i in xrange(nrestarts):
                (o, _)     = action.take(task)
                n          = indices[o]
                counts[n] += 1

            yield counts.tolist()

    return {
        "action"   : jsonify(action),
        "outcomes" : [jsonify(u) for u in action.outcomes],
        "counts"   : list(yield_counts()),
        }

@with_flags_parsed(
    usage = "usage: %prog [options]",
    )
def main(positional):
    """
    Main.
    """

    flags = module_flags.given

    # load configuration
    solvers = get_named_solvers()

    with open(flags.solvers) as file:
        solvers = [solvers[s] for s in json.load(file)]

    with open(flags.cutoffs) as file:
        cutoffs = [TimeDelta(seconds = s) for s in json.load(file)]

    with open(flags.tasks) as file:
        tasks = [SAT_WorldTask(join(getenv("EXPERIMENT_ROOT"), "tasks", p), p) for p in json.load(file)]

    # get training data
    actions = [SAT_WorldAction(s, c) for (s, c) in product(solvers, cutoffs)]
    data    = {
        "tasks"   : [jsonify(t) for t in tasks],
        "history" : [practice(a, tasks, flags.restarts) for a in actions],
        }

    # and store it
    with open(flags.output, "w") as file:
        json.dump(data, file)

