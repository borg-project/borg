"""
utexas/papers/nips2009/evaluees.py

Evaluate selection strategies.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy

from functools import partial
from collections import defaultdict
from utexas.alog import DefaultLogger
from utexas.flags import (
    Flag,
    FlagSet,
    )
from utexas.statistics.dcm import DCM_Estimator
from utexas.statistics.mixture import (
    FixedIndicatorMixtureEstimator,
    ExpectationMaximizationMixtureEstimator,
    )
from utexas.statistics.multinomial import MultinomialEstimator
from utexas.papers.nips2009.world import WorldDescription
from utexas.papers.nips2009.models import (
    OracleActionModel,
    RandomActionModel,
    MultinomialActionModel,
    DCM_MixtureActionModel,
    MultinomialMixtureActionModel,
    )
from utexas.papers.nips2009.planners import (
    HardMyopicActionPlanner,
    SoftMyopicActionPlanner,
    )
from utexas.papers.nips2009.strategies import (
    FixedSelectionStrategy,
    ModelingSelectionStrategy,
    )

log = get_logger(__name__)

class Evaluee(object):
    """
    A portfolio strategy under evaluation.
    """

    def __init__(self, name, strategy, world):
        """
        Initialize.
        """

        self.name = name
        self.strategy = strategy
        self.world = world
        self.total_utility = 0.0
        self.total_max_utility = 0.0
        self.spent = 0.0
        self.nsolved = 0
        self.action_log = numpy.zeros(world.nactions, dtype = numpy.uint32)

    @property
    def average_utility(self):
        """
        Return the average per-restart utility accrued.
        """

        return self.total_utility / self.spent

class NamedEvalueeFactory(object):
    """
    Construct standard evaluees.
    """

    class Flags(FlagSet):
        """
        Flags for the containing module.
        """

        flag_set_title = "Evaluee Configuration"

        planner_flag = \
            Flag(
                "--planner",
                default = "hardmax",
                choices = ("softmax", "hardmax"),
                help = "use planner NAME [%default]",
                metavar = "NAME",
                )

    def __init__(self, world, training_history, ncomponents, discount_factor, flags = Flags.given):
        """
        Initialize.
        """

        self.world = world
        self.training_history = training_history
        self.ncomponents = ncomponents
        self.discount_factor = discount_factor
        self.flags = flags

        # set up constructor names
        self.constructors = {
            "oracle": self.make_oracle,
            "random": self.make_random,
            "multinomial": self.make_multinomial,
            "multinomial_mixture": self.make_multinomial_mixture,
            "multinomial_fixed_mixture": self.make_multinomial_fixed_mixture,
            "dcm_mixture": self.make_dcm_mixture,
            "dcm_fixed_mixture": self.make_dcm_fixed_mixture,
            }

        for action in self.world.actions:
            self.constructors[str(action)] = partial(self.make_fixed, action = action)

        self.name_classes = {
            "fixed": [str(a) for a in self.world.actions],
            }

        self.planner_types = {
            "hardmax": HardMyopicActionPlanner,
            "softmax": SoftMyopicActionPlanner,
            }

        self.planner = self.planner_types[self.flags.planner](self.world, self.discount_factor)

    def expand(self, name):
        """
        Expand names of evaluee classes into their members.

        The name "fixed", for example, becomes the list of names of fixed
        solvers; names of individual evaluees expand to singleton tuples.
        """

        try:
            return self.name_classes[name]
        except KeyError:
            self.constructors[name]

            return [name]

    def make(self, name):
        """
        Build a new evaluee.
        """

        return self.constructors[name](name)

    def make_oracle(self, name):
        """
        Return a new oracle strategy for evaluation.
        """

        log.info("building oracle model")

        # build strategy
        model = OracleActionModel(self.world)
        strategy = ModelingSelectionStrategy(self.world, model, self.planner)

        # done
        return Evaluee(name, strategy, self.world)

    def make_random(self, name):
        """
        Return a new random-model strategy for evaluation.
        """

        log.info("building random model")

        # build strategy
        model = RandomActionModel(self.world)
        strategy = ModelingSelectionStrategy(self.world, model, self.planner)

        # done
        return Evaluee(name, strategy, self.world)

    def make_multinomial(self, name):
        """
        Return a new multinomial strategy for evaluation.
        """

        log.info("building multinomial model")

        # build strategy
        model = MultinomialActionModel(self.world, self.training_history)
        strategy = ModelingSelectionStrategy(self.world, model, self.planner)

        # informative output
        log.info("multinomial parameters follow:")

        for action in self.world.actions:
            log.info("%s: %s" % (action, model.outcome_probabilities[action.n]))

        # done
        return Evaluee(name, strategy, self.world)

    def make_dcm_mixture(self, name):
        """
        Return a new DCM mixture strategy for evaluation.
        """

        log.info("building DCM mixture model; parameters will follow")

        # build strategy
        component_estimators = [[DCM_Estimator()] * self.ncomponents] * self.world.nactions
        estimator = ExpectationMaximizationMixtureEstimator(component_estimators)
        model = DCM_MixtureActionModel(self.world, self.training_history, estimator)
        strategy = ModelingSelectionStrategy(self.world, model, self.planner)

        # informative output
        log.info("DCM mixture parameters follow:")

        for action in self.world.actions:
            components = model.mixture.components[action.n]

            for component in components:
                log.info("%s: %s * %f" % (action, component.mean, component.burstiness))

        # done
        return Evaluee(name, strategy, self.world)

    def make_multinomial_mixture(self, name):
        """
        Return a new multinomial mixture strategy for evaluation.
        """

        log.info("building multinomial mixture model")

        # build strategy
        component_estimators = [[MultinomialEstimator()] * self.ncomponents] * self.world.nactions
        estimator = ExpectationMaximizationMixtureEstimator(component_estimators)
        model = MultinomialMixtureActionModel(self.world, self.training_history, estimator)
        strategy = ModelingSelectionStrategy(self.world, model, self.planner)

        # informative output
        log.info("multinomial mixture parameters follow:")

        for action in self.world.actions:
            components = model.mixture.components[action.n]

            for component in components:
                log.info("%s: %s" % (action, component.mean))

        # done
        return Evaluee(name, strategy, self.world)

    def make_multinomial_fixed_mixture(self, name):
        """
        Return a new multinomial "task-class equivalence" mixture strategy for evaluation.
        """

        log.info("building multinomial equivalence mixture model")

        # build strategy
        ntasks_train = self.training_history.get_positive_counts().shape[0]
        component_estimators = [[MultinomialEstimator()] * ntasks_train] * self.world.nactions
        estimator = FixedIndicatorMixtureEstimator(component_estimators)
        model = MultinomialMixtureActionModel(self.world, self.training_history, estimator)
        strategy = ModelingSelectionStrategy(self.world, model, self.planner)

        # informative output
        log.info("multinomial mixture parameters follow:")

        for k in xrange(model.mixture.components.shape[1]):
            def expectation(action):
                return numpy.sum(model.mixture.components[action.n, k].beta * self.world.utilities)

            parameters = self.__action_matrix(expectation)

            log.info("conmponent %i:\n%s", k, parameters)

        # done
        return Evaluee(name, strategy, self.world)

    def make_dcm_fixed_mixture(self, name):
        """
        Return a new DCM "task-class equivalence" mixture strategy for evaluation.
        """

        log.info("building DCM equivalence mixture model")

        # build strategy
        ntasks_train = self.training_history.get_positive_counts().shape[0]
        component_estimators = [[DCM_Estimator()] * ntasks_train] * self.world.nactions
        estimator = FixedIndicatorMixtureEstimator(component_estimators)
        model = DCM_MixtureActionModel(self.world, self.training_history, estimator)
        strategy = ModelingSelectionStrategy(self.world, model, self.planner)

        # informative output
#        log.info("DCM mixture parameters follow:")

#        for k in xrange(model.mixture.components.shape[1]):
#            def expectation(action):
#                return numpy.sum(model.mixture.components[action.n, k].beta * self.world.utilities)

#            parameters = self.__action_matrix(expectation)

#            log.info("conmponent %i:\n%s", k, parameters)

        # done
        return Evaluee(name, strategy, self.world)

    def make_fixed(self, name, action):
        """
        Return a new fixed strategy for evaluation.
        """

        # done
        return Evaluee(name, FixedSelectionStrategy(action), self.world)

    def __action_matrix(self, m):
        """
        Return a sorted matrix of mapped action values.
        """

        actions_by_cutoff = defaultdict(list)

        for action in self.world.actions:
            actions_by_cutoff[action.cutoff].append(action)

        for actions in actions_by_cutoff.itervalues():
            actions.sort(key = lambda a: a.solver_name)

        sorted_rows = sorted(actions_by_cutoff.iteritems(), key = lambda (c, a): c)
        utility_rows = [[m(a) for a in aa] for (_, aa) in sorted_rows]

        return numpy.array(utility_rows)

