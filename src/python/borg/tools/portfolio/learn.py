"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.portfolio.learn import main

    raise SystemExit(main())

from cargo.log import get_logger

log = get_logger(__name__)

def build_request(request, trainer):
    """
    Build a portfolio object as requested.
    """

    from borg.solvers          import PortfolioSolver
    from borg.portfolio.models import build_model

    builders = {
        "solver" : PortfolioSolver.build,
        "model"  : build_model,
        }

    (_, kind) = request.keys()

    return builders[kind](request[kind], trainer)

def main():
    """
    Script entry point.
    """

    # get command line arguments
    import borg.data

    from cargo.sql.alchemy import SQL_Engines
    from cargo.json        import load_json
    from cargo.flags       import parse_given

    (train_uuids_path, request_path, solver_path) = \
        parse_given(
            usage = "%prog <train_uuids.json> <request.json> <out.pickle> [options]",
            )

    train_uuids = load_json(train_uuids_path)
    request     = load_json(request_path)

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine",        level = "WARNING")
    get_logger("cargo.statistics.mixture", level = "DETAIL")
    get_logger("borg.portfolio.sat_world", level = "DETAIL")

    # construct the solver
    from cargo.sql.alchemy    import make_session
    from borg.data            import research_connect
    from borg.portfolio.world import build_trainer

    ResearchSession = make_session(bind = research_connect())
    trainer         = build_trainer(request["domain"], train_uuids, ResearchSession)
    requested       = build_request(request, trainer)

    # write it to disk
    import cPickle as pickle

    with open(solver_path, "w") as file:
        pickle.dump(requested, file, -1)

