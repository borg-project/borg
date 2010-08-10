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

    # need a place to dump profiling results
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile() as named:
        # profile the computation
        from cProfile                         import runctx
        from borg.portfolio.bellman           import compute_bellman_plan
        from borg.portfolio.test.test_bellman import build_real_model

        context = {"c": compute_bellman_plan, "m": build_real_model()}
        profile = runctx("c(m, 6, 1e6, 1.0)", {}, context, named.name)

        # extract the results
        from pstats import Stats

        stats = Stats(named.name)

    # display a report
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()

