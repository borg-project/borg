"""
utexas/papers/nips2009/evaluate.py

Evaluate selection strategies.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import sys
import numpy
import tables
import utexas.flags

from itertools import product
from contextlib import closing
from tables import NoSuchNodeError
from utexas.kit import (
    random_subsets,
    IntRanges,
    FloatRanges,
    )
from utexas.alog import DefaultLogger
from utexas.flags import (
    Flag,
    FlagSet,
    )
from utexas.tables import (
    goc_table,
    goc_earray,
    )
from utexas.papers.nips2009.hdf import (
    WorldActionsTableDescription,
    SAT_EvaluationsTableDescription,
    )
from utexas.papers.nips2009.world import WorldDescription
from utexas.papers.nips2009.evaluees import NIPS2009_EvalueeFactory
from utexas.papers.nips2009.strategies import ActionHistory

log = DefaultLogger(__name__)

class TestEnvironment(object):
    """
    Evaluate algorithm selection strategies.
    """

    def __init__(self, world, test_tasks, task_time):
        """
        Initialize.
        """

        # parameters
        self.world = world
        self.test_tasks = test_tasks
        self.task_time = task_time

    def evaluate(self, evaluee):
        """
        Evaluate the specified evaluee.
        """

        log.info("evaluating %s", evaluee.name)

        for (ntest, task) in enumerate(self.test_tasks):
            log.info("evaluating on task %i (test %i of %i)", task.n, ntest + 1, len(self.test_tasks))
            log.info("task has path %s", task.path)

            self.evaluate_on(evaluee, task)

            log.info(
                "total max utility for %s: %.2f",
                evaluee.name,
                evaluee.total_max_utility,
                )

    def evaluate_on(self, evaluee, task):
        """
        Evaluate on a specific task.
        """

        best_utility = 0.0
        remaining = self.task_time

        while remaining > 0.0:
            # let the evaluee take an action
            (outcome, spent) = self.evaluate_once_on(evaluee, task, remaining)

            # deal with that action's outcome
            remaining -= spent

            if outcome.utility > best_utility:
                best_utility = outcome.utility
            if outcome.n == self.world.nsuccess:
                evaluee.nsolved += 1

                break

        evaluee.total_max_utility += best_utility

        log.info("%s had max utility %.2f with %.1fs remaining", evaluee.name, best_utility, remaining)

    def evaluate_once_on(self, evaluee, task, remaining):
        """
        Evaluate once on a specific task.
        """

        # let the evaluee select an action
        actions = [a for a in self.world.actions if a.cutoff <= remaining]
        action_generator = evaluee.strategy.select(task, actions)
        action = action_generator.send(None)

        assert action in actions

        # take that action
        outcome = self.world.sample_action(task, action)

        log.info("%.1fs: [%i] %s -> %s", remaining, task.n, action, outcome)

        try:
            action_generator.send(outcome)
        except StopIteration:
            pass

        evaluee.total_utility += outcome.utility
        evaluee.spent += action.cutoff
        evaluee.action_log[action.n] += 1

        return (outcome, action.cutoff)

class Evaluation(object):
    """
    Evaluate algorithm selection strategies.
    """

    class Flags(FlagSet):
        """
        Flags for the containing class.
        """

        flag_set_title = "Strategy Testing"

        evaluation_file_flag = \
            Flag(
                "--evaluation-file",
                default = "evaluation.h5",
                metavar = "PATH",
                help = "write results to PATH [%default]",
                )
        ntasks_test_flag = \
            Flag(
                "--ntasks-test",
                default = 128,
                type = int,
                metavar = "N",
                help = "use N test tasks [%default]",
                )
        nntasks_train_flag = \
            Flag(
                "--nntasks-train",
                default = IntRanges.only(32),
                type = IntRanges,
                metavar = "NN",
                help = "use NN training tasks [%default]",
                )
        nnrestarts_train_flag = \
            Flag(
                "--nnrestarts-train",
                default = IntRanges.only(32),
                type = IntRanges,
                metavar = "NN",
                help = "use NN training actions [%default]",
                )
        train_split_seed_flag = \
            Flag(
                "--train-split-seed",
                default = None,
                type = int,
                metavar = "N",
                help = "use seed N for train/test sampling",
                )
        nncomponents_flag = \
            Flag(
                "--nncomponents",
                default = IntRanges.only(8),
                type = IntRanges,
                metavar = "NN",
                help = "use NN mixture components [%default]",
                )
        discount_factor_flag = \
            Flag(
                "--discount-factor",
                default = FloatRanges.only(0.99),
                type = FloatRanges,
                help = "use discount factor XX [%default]",
                metavar = "XX",
                )
        task_time_flag = \
            Flag(
                "--task-time",
                default = FloatRanges.only(512.0),
                type = FloatRanges,
                help = "allow XX seconds per test task [%default]",
                metavar = "XX",
                )
        nruns_flag = \
            Flag(
                "--nruns",
                default = 1,
                type = int,
                metavar = "N",
                help = "run each setting N times [%default]",
                )
        evaluee_flag = \
            Flag(
                "-e",
                "--evaluee",
                dest = "evaluees",
                default = [],
                action = "append",
                metavar = "NAME",
                help = "evaluate strategy NAME",
                )

    def __init__(self, flags = None):
        """
        Initialize.
        """

        # parameters
        self.world = WorldDescription()
        self.flags = flags or Evaluation.Flags.given

        # desribe the world
        log.info("the world has the following %i actions:", len(self.world.actions))

        for action in self.world.actions:
            log.info("%s", action)

    def __build_train_test_split(self, ntasks_train, nrestarts_train):
        """
        Evaluate the strategy set.
        """

        log.info(
            "sampling (%i*%i)/%i train/test split",
            ntasks_train,
            nrestarts_train,
            self.flags.ntasks_test,
            )

        if self.flags.train_split_seed is None:
            random = numpy.random
        else:
            random = RandomState(seed = self.flags.train_split_seed)

        (train_tasks, test_tasks) = \
            random_subsets(
                self.world.tasks,
                (ntasks_train, self.flags.ntasks_test),
                random = random,
                )
        training_history = \
            ActionHistory.sample(
                self.world,
                train_tasks,
                nrestarts_train,
                random = random,
                )

        return (test_tasks, training_history)

    def __evaluate(self, evaluation_file):
        """
        Run the evaluation.
        """

        # store actions or verify consistency
        world_actions_table = goc_table(evaluation_file, "/world_actions", WorldActionsTableDescription)

        if world_actions_table.nrows > 0:
            for action in self.world.actions:
                row = world_actions_table[action.n]

                assert row["solver"] == action.nsolver
                assert row["cutoff"] == action.cutoff
        else:
            row = world_actions_table.row

            for action in self.world.actions:
                row["solver"] = action.nsolver
                row["cutoff"] = action.cutoff

                row.append()

            world_actions_table.flush()

        # run the evaluation
        over = \
            product(
                xrange(self.flags.nruns),
                self.flags.nntasks_train,
                self.flags.nnrestarts_train,
                self.flags.nncomponents,
                self.flags.discount_factor,
                self.flags.task_time,
                )

        for parameters in over:
            # another evaluation
            log.info("beginning an evaluation %s", str(parameters))

            (nrun, ntasks_train, nrestarts_train, ncomponents, discount_factor, task_time) = parameters

            # generate a new train/test split, then build the factory
            (test_tasks, training_history) = self.__build_train_test_split(ntasks_train, nrestarts_train)

            log.info("modeling with %i mixture components", ncomponents)

            factory = NIPS2009_EvalueeFactory(self.world, training_history, ncomponents, discount_factor)
            evaluee_names = sum((factory.expand(name) for name in self.flags.evaluees), [])

            for evaluee_name in evaluee_names:
                # build and test the evaluee
                evaluee = factory.make(evaluee_name)
                test_environment = TestEnvironment(self.world, test_tasks, task_time)

                test_environment.evaluate(evaluee)

                # record its performance
                evaluation_table = \
                    goc_table(
                        evaluation_file,
                        "/%s/evaluations/%s" % (factory.flags.planner, evaluee_name),
                        SAT_EvaluationsTableDescription,
                        )
                row = evaluation_table.row

                row["tasks_test"] = self.flags.ntasks_test
                row["task_time"] = task_time
                row["tasks_train"] = ntasks_train
                row["restarts_train"] = nrestarts_train
                row["components"] = ncomponents
                row["solved"] = evaluee.nsolved
                row["spent"] = evaluee.spent
                row["discount"] = discount_factor

                row.append()

                # record its action counts
                action_log_earray = \
                    goc_earray(
                        evaluation_file,
                        "/%s/action_logs/%s" % (factory.flags.planner, evaluee_name),
                        tables.Int32Atom(),
                        (0, self.world.nactions),
                        )

                action_log_earray.append([evaluee.action_log])

                # and straight on 'til morning
                evaluation_table.flush()
                action_log_earray.flush()

    def evaluate(self):
        """
        Run the evaluation.
        """

        # set up the evaluation file
        evaluation_file = tables.openFile(self.flags.evaluation_file, "a")

        with closing(evaluation_file):
            self.__evaluate(evaluation_file)

# invocation!
if __name__ == "__main__":
    log.info("arguments: %s", sys.argv)

    utexas.flags.parse_given()

    Evaluation().evaluate()

