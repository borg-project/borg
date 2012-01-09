"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import nose.tools
import numpy
import borg

worlds = {
    "short": (
        numpy.log([[[1.0, 0.2, 0.2, 0.19]]]),
        numpy.log([1.0]),
        [(0, 1), (0, 1)]
        ),
    "long": (
        numpy.log([[[1.0, 0.2, 0.2, 0.01]]]),
        numpy.log([1.0]),
        [(0, 3)],
        ),
    "2worlds": (
        numpy.log([
            [[1.0, 1.0, 0.5, 0.4]],
            [[0.2, 0.2, 0.2, 0.2]],
            ]),
        numpy.log([0.5, 0.5]),
        [(0, 0), (0, 2)],
        ),
    "micro": (
        numpy.log([[[0.99, 0.989, 0.989, 0.989]]]),
        numpy.log([1.0]),
        [(0, 0), (0, 0), (0, 1)],
        ),
    }

def assert_planner_ok(planner, world_name):
    (survival, weights, expected) = worlds[world_name]

    plan = planner.plan(survival, weights)

    nose.tools.assert_equal(sorted(plan), expected)

def test_knapsack_planner():
    planner = borg.planners.KnapsackPlanner()

    def assert_knapsack_planner_ok(world_name):
        assert_planner_ok(planner, world_name)

    yield (assert_knapsack_planner_ok, "short")
    yield (assert_knapsack_planner_ok, "long")
    yield (assert_knapsack_planner_ok, "2worlds")

def test_max_length_knapsack_planner():
    planner = borg.planners.MaxLengthKnapsackPlanner(3)

    def assert_this_planner_ok(world_name):
        assert_planner_ok(planner, world_name)

    yield (assert_this_planner_ok, "short")
    yield (assert_this_planner_ok, "long")
    yield (assert_this_planner_ok, "2worlds")
    yield (assert_this_planner_ok, "micro")

def test_streeter_planner():
    planner = borg.planners.StreeterPlanner()

    def assert_streeter_planner_ok(world_name):
        assert_planner_ok(planner, world_name)

    yield (assert_streeter_planner_ok, "short")
    #yield (assert_streeter_planner_ok, "long")
    yield (assert_streeter_planner_ok, "2worlds")

def test_bellman_planner():
    planner = borg.planners.BellmanPlanner()

    def assert_bellman_planner_ok(world_name):
        assert_planner_ok(planner, world_name)

    yield (assert_bellman_planner_ok, "short")
    yield (assert_bellman_planner_ok, "long")
    yield (assert_bellman_planner_ok, "2worlds")

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

