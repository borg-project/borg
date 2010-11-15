"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.bench_bellman import main

    raise SystemExit(main())

def main():
    """
    Benchmark the Bellman plan computation code.
    """

    from borg.portfolio.bellman           import compute_bellman_plan
    from borg.portfolio.test.test_bellman import build_real_model

    model = build_real_model()
    stats = call_profiled(lambda: compute_bellman_plan(model, 6, 1e6, 1.0))

    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()

