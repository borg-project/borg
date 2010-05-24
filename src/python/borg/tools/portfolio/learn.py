# vim: set fileencoding=UTF-8 :
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.portfolio.learn import main

    raise SystemExit(main())

import numpy

from cargo.log import get_logger

log = get_logger(__name__)

def build_requested(request, trainer):
    """
    Build a solver according to a configuration request.
    """

    from borg.solvers import PortfolioSolver

    return PortfolioSolver.build(request, trainer)

def main():
    """
    Main.
    """

    # get command line arguments
    import borg.data

    from cargo.sql.alchemy import SQL_Engines
    from cargo.json        import load_json
    from cargo.flags       import parse_given

    (request_path, solver_path) = \
        parse_given(
            usage = "%prog <request.json> <out.pickle> [options]",
            )

    request = load_json(request_path)

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("cargo.statistics.mixture", level = "DETAIL")

    # construct the builder
    from cargo.sql.alchemy        import make_session
    from borg.data                import research_connect
    from borg.portfolio.sat_world import SAT_Trainer

    ResearchSession = make_session(bind = research_connect())
    trainer         = SAT_Trainer(ResearchSession)
    solver          = build_requested(request, trainer)

    # write it to disk
    with open(solver_path, "w") as file:
        pickle.dump(solver, file, -1)

