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

def test_bellman_planner_multi():
    plan = \
        borg.planners.BellmanPlanner().plan(
            numpy.log([
                [[1.0, 1.0, 0.5, 0.4]],
                [[0.2, 0.2, 0.2, 0.2]],
                ]),
            numpy.log([0.5, 0.5]),
            )

    nose.tools.assert_equal(plan, [(0, 0), (0, 2)])

