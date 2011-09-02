"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import nose.tools
import numpy
import borg

def test_bellman_planner_short():
    plan = \
        borg.planners.BellmanPlanner().plan(
            numpy.log([[[1.0, 0.2, 0.2, 0.19]]]),
            numpy.log([1.0]),
            )

    nose.tools.assert_equal(plan, [(0, 1), (0, 1)])

def test_bellman_planner_long():
    plan = \
        borg.planners.BellmanPlanner().plan(
            numpy.log([[[1.0, 0.2, 0.2, 0.01]]]),
            numpy.log([1.0]),
            )

    nose.tools.assert_equal(plan, [(0, 3)])

def test_bellman_planner_2worlds():
    plan = \
        borg.planners.BellmanPlanner().plan(
            numpy.log([
                [[1.0, 1.0, 0.5, 0.4]],
                [[0.2, 0.2, 0.2, 0.2]],
                ]),
            numpy.log([0.5, 0.5]),
            )

    nose.tools.assert_equal(sorted(plan), [(0, 0), (0, 2)])

def test_bellman_planner_2solvers():
    plan = \
        borg.planners.BellmanPlanner().plan(
            numpy.log([
                [[1.0, 0.2, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [1.0, 0.2, 0.1, 0.1]],
                ]),
            numpy.log([0.5, 0.5]),
            )

    nose.tools.assert_equal(sorted(plan), [(0, 1), (1, 1)])

def test_bellman_planner_impossibilities():
    plan = \
        borg.planners.BellmanPlanner().plan(
            [
                [[0.0, 0.0, -numpy.inf, -numpy.inf], [0.0, 0.0, 0.0, -0.2]],
                [[0.0, 0.0, 0.0, -0.1], [-0.2, -0.2, -0.2, -0.2]],
                ],
            numpy.log([0.5, 0.5]),
            )

    nose.tools.assert_equal(sorted(plan), [(0, 2), (1, 0)])

def test_bellman_planner_replan():
    model0 = \
        borg.models.MultinomialModel(
            10.0,
            numpy.log([
                [[1.0, 0.2, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [1.0, 0.2, 0.1, 0.1]],
                ]),
            numpy.log([0.5, 0.5]),
            )
    plan0 = borg.planners.BellmanPlanner().plan(model0.log_survival, model0.log_weights)
    print "plan0:", plan0
    model1 = model0.condition(plan0[:1])
    print "model1.log_weights:", model1.log_weights
    plan1 = borg.planners.BellmanPlanner().plan(model1.log_survival[..., :-plan0[0][1] - 1], model1.log_weights)
    print "plan1:", plan1

    nose.tools.assert_equal(plan1, plan0[1:])

