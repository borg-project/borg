"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import abstractmethod
from cargo.log   import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

#class ModelingStrategy(Strategy):
    #"""
    #A strategy that employs a model of its actions.
    #"""

    #def __init__(self, model, planner):
        #"""
        #Initialize.
        #"""

        #import numpy

        #dimensions = (len(model.actions), max(len(a.outcomes) for a in model.actions))

        #self._model   = model
        #self._planner = planner
        #self._history = numpy.zeros(dimensions, numpy.uint)

    #def reset(self):
        #"""
        #Prepare to solve a new task.
        #"""

        #self._history[:, :] = 0

    #def see(self, action, outcome):
        #"""
        #Witness the outcome of an action.
        #"""

        #self._history[self._model.actions.index(action), action.outcomes.index(outcome)] += 1

    #def choose(self, budget, random):
        #"""
        #Return the selected action.
        #"""

        #return self._planner.select(self._model, self._history, budget, random)

#class BellmanStrategy(SequenceStrategy):
    #"""
    #A strategy that employs a model of its actions.
    #"""

    #def __init__(self, model, horizon, budget, discount = 1.0):
        #"""
        #Initialize.
        #"""

        #from datetime                      import timedelta
        #from borg.portfolio.bellman        import compute_bellman_plan
        #from borg.portfolio.decision_world import SolverAction

        #plan = compute_bellman_plan(model, horizon, budget, discount)

        #plan[-1] = SolverAction(plan[-1].solver, timedelta(seconds = 1e6))

        #log.info("Bellman plan follows (horizon %i, budget %f)", horizon, budget)

        #for (i, action) in enumerate(plan):
            #log.info("action %i: %s", i, action.description)

        #SequenceStrategy.__init__(self, plan)

#class ChainedStrategy(Strategy):
    #"""
    #Employ a sequence of strategies.
    #"""

    #def __init__(self, strategies):
        #"""
        #Initialize.
        #"""

        #self._strategies = strategies

    #def _yield_choices(self):
        #"""
        #Iterate over this strategy's choices.
        #"""

        #(budget, random) = yield

        #for strategy in self._strategies:
            #while True:
                #chosen = strategy.choose(budget, random)

                #if chosen is None:
                    #break
                #else:
                    #(budget, random) = yield chosen

        #while True:
            #yield None

    #def reset(self):
        #"""
        #Prepare to solve a new task.
        #"""

        ## reset the substrategies
        #for strategy in self._strategies:
            #strategy.reset()

        ## reset our iteration
        #self._iterable = self._yield_choices()

        #self._iterable.next()

    #def see(self, action, outcome):
        #"""
        #Witness the outcome of an action.
        #"""

        #for strategy in self._strategies:
            #strategy.see(action, outcome)

    #def choose(self, budget, random):
        #"""
        #Return the selected action.
        #"""

        #return self._iterable.send((budget, random))

