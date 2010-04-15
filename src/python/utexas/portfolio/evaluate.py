"""
utexas/portfolio/evaluate.py

Evaluate selection strategies.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import sys
import numpy

from itertools       import product
from contextlib      import closing
from cargo.flags     import (
    Flag,
    FlagSet,
    IntRanges,
    FloatRanges,
    )
from cargo.log       import get_logger
from cargo.temporal  import TimeDelta
from cargo.iterators import random_subsets

log = get_logger(__name__)

class PortfolioTestScore(object):
    """
    The result of a portfolio test.
    """

    def __init__(self, world):
        """
        Initialize.
        """

        self.total_utility     = 0.0
        self.total_max_utility = 0.0
        self.total_discounted  = 0.0
        self.spent             = TimeDelta()
        self.nsolved           = 0
        self.action_log        = numpy.zeros(world.nactions, dtype = numpy.uint32)

class PortfolioTest(object):
    """
    Evaluate algorithm selection strategies.
    """

    def __init__(self, world, test_tasks, task_time, discount, actions = None, success = 1.0):
        """
        Initialize.
        """

        self.world      = world
        self.test_tasks = test_tasks
        self.task_time  = task_time
        self.score      = PortfolioTestScore(world)
        self.discount   = discount

        if actions is None:
            actions = world.actions

        self.actions = actions
        self.success = success

    def evaluate(self, strategy):
        """
        Evaluate the specified evaluee.
        """

        log.note("running a portfolio evaluation")

        for (ntest, task) in enumerate(self.test_tasks):
            log.info("evaluating on task %i (test %i of %i)", task.n, ntest + 1, len(self.test_tasks))
            log.debug("task has uuid %s", task.task.uuid)

            self.evaluate_on(strategy, task)

        log.detail(
            "total max utility for %s: %.2f",
            strategy,
            self.score.total_max_utility,
            )

        return self.score

    def evaluate_on(self, strategy, task):
        """
        Evaluate on a specific task.
        """

        best_utility = 0.0
        remaining    = self.task_time
        ntaken       = 0
        total_spent  = TimeDelta()

        # FIXME remove the arbitrary action limit
        while remaining > TimeDelta() and ntaken < 50:
            # let the evaluee take an action
            (outcome, spent) = self.evaluate_once_on(strategy, task, remaining)

            if spent is None:
                break

            ntaken      += 1
            total_spent += spent

            # deal with that action's outcome
            remaining -= spent

            if outcome.utility > best_utility:
                best_utility = outcome.utility
            if outcome.utility >= self.success:
                self.score.nsolved += 1
                self.score.spent   += total_spent

                break

        self.score.total_max_utility += best_utility

        log.info("strategy had max utility %.2f with %s remaining", best_utility, remaining)

    def evaluate_once_on(self, strategy, task, remaining):
        """
        Evaluate once on a specific task.
        """

        # let the evaluee select an action
        actions = [a for a in self.actions if a.cutoff <= remaining]

        if not actions:
            return (None, None)

        action_generator = strategy.select(task, actions)
        action           = action_generator.send(None)

        if action is None:
            return (None, None)

        assert action in actions

        # take that action
        (outcome, spent) = self.world.act_once_extra(task, action)

        log.detail("%s: [%i] %s -> %s (%s)", remaining, task.n, action, outcome, spent)

        try:
            action_generator.send(outcome)
        except StopIteration:
            pass

        self.score.total_utility        += outcome.utility
        self.score.action_log[action.n] += 1
        self.score.total_discounted     += outcome.utility * self.discount**action.cutoff.as_s # FIXME broken if nonzero non-success actions exist

        return (outcome, spent)

def build_train_test_split(world, ntasks_train, nrestarts_train, ntasks_test, random = numpy.random):
    """
    Evaluate the strategy set.
    """

    log.info(
        "sampling (%i*%i)/%i train/test split",
        ntasks_train,
        nrestarts_train,
        ntasks_test,
        )

    (train_tasks, test_tasks) = \
        random_subsets(
            world.tasks,
            (ntasks_train, ntasks_test),
            random = random,
            )
    training_history = \
        world.act_all(
            train_tasks,
            world.actions,
            nrestarts_train,
            random = random,
            )

    return (test_tasks, training_history)

