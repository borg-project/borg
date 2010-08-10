"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def build_real_model():
    """
    Build a reasonably-representative test model.
    """

    # set up the model
    import numpy

    from cargo.temporal           import TimeDelta
    from cargo.statistics.dcm     import DirichletCompoundMultinomial as DCM
    from cargo.statistics.tuple   import TupleDistribution as TD
    from cargo.statistics.mixture import FiniteMixture
    from borg.solvers             import LookupSolver
    from borg.portfolio.world     import SolverAction
    from borg.portfolio.models    import DistributionModel

    actions = [
        SolverAction(LookupSolver("foo"), TimeDelta(seconds = 1.0)),
        SolverAction(LookupSolver("bar"), TimeDelta(seconds = 1.0)),
        SolverAction(LookupSolver("baz"), TimeDelta(seconds = 1.0)),
        SolverAction(LookupSolver("qux"), TimeDelta(seconds = 1.0)),
        ]
    components = [
        TD(map(DCM, [[0.10, 0.90], [0.20, 0.80], [0.15, 0.85], [0.05, 0.95]])),
        TD(map(DCM, [[0.10, 0.90], [0.20, 0.80], [0.15, 0.85], [0.05, 0.95]])),
        TD(map(DCM, [[0.10, 0.90], [0.20, 0.80], [0.15, 0.85], [0.05, 0.95]])),
        TD(map(DCM, [[0.10, 0.90], [0.20, 0.80], [0.15, 0.85], [0.05, 0.95]])),
        ]
    mixture = FiniteMixture([0.25] * 4, components)

    return DistributionModel(mixture, actions)

def test_compute_bellman():
    """
    Test Bellman-optimal planning.
    """

    # test the computation
    from nose.tools             import (
        assert_equal,
        assert_almost_equal,
        )
    from borg.portfolio.bellman import compute_bellman_plan

    (expectation, plan) = compute_bellman_plan(build_real_model(), 2, 1e6, 0.9)

    assert_almost_equal(expectation, 0.344)
    assert_equal([a.solver.name for a in plan], ["bar", "bar"])

