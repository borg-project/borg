"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.portfolio.learn import main

    raise SystemExit(main())

from cargo.log import get_logger

log = get_logger(__name__)

def main():
    """
    Script entry point.
    """

    # get command line arguments
    import borg.data

    from uuid        import UUID
    from cargo.json  import load_json
    from cargo.flags import parse_given

    (train_uuids_path, request_path, solver_path) = \
        parse_given(
            usage = "%prog <uuids.json> <request.json> <out.pickle> [options]",
            )

    train_uuids = map(UUID, load_json(train_uuids_path))
    request     = load_json(request_path)

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine",         level = "WARNING")
    get_logger("cargo.statistics.mixture",  level = "DETAIL")
    get_logger("borg.solvers.portfolio",    level = "DETAIL")
    get_logger("borg.portfolio.bellman",    level = "DETAIL")
    get_logger("borg.portfolio._bellman",   level = "DETAIL")
    get_logger("borg.portfolio.sat_world",  level = "DETAIL")
    get_logger("borg.portfolio.strategies", level = "DETAIL")

    # construct the solver
    from cargo.sql.alchemy    import make_session
    from borg.data            import research_connect
    from borg.portfolio.world import Trainer
    from borg.solvers         import AbstractSolver

    ResearchSession = make_session(bind = research_connect())
    trainer         = Trainer.build(ResearchSession, train_uuids, request["trainer"])
    requested       = AbstractSolver.build(trainer, request["solver"])

    # write it to disk
    import cPickle as pickle

    with open(solver_path, "w") as file:
        pickle.dump(requested, file, -1)

