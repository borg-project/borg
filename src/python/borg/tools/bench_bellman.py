"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.bench_bellman import main

    raise SystemExit(main())

def benchmark_bellman():
    """
    Benchmark computation of an optimal plan.
    """

    # compute the plan
    from time                             import clock
    from borg.portfolio.bellman           import compute_bellman_plan
    from borg.portfolio.test.test_bellman import build_real_model

    model  = build_real_model()
    before = clock()

    (expectation, plan) = compute_bellman_plan(model, 6, 1e6, 1.0)

    print "elapsed: %.2f seconds" % (clock() - before)
    print "expectation: %f"       % expectation
    print "plan: %s"              % [a.solver.name for a in plan]

def main():
    """
    Benchmark the Bellman plan computation code.
    """

    benchmark_bellman()

