"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                    import call
    from borg.tools.build_solver import main

    call(main)

from plac      import annotations
from cargo.log import get_logger
from borg      import defaults

log = get_logger(__name__)

@annotations(
    url = ("database URL", "option"),
    )
def main(train_uuids_path, solver_py, out_path, url = defaults.research_url):
    """
    Script entry point.
    """

    # transform arguments
    from uuid        import UUID
    from cargo.json  import load_json

    train_uuids = map(UUID, load_json(train_uuids_path))

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine"        , level = "WARNING")
    get_logger("cargo.statistics.mixture" , level = "DETAIL")
    get_logger("borg.solvers.portfolio"   , level = "DETAIL")
    get_logger("borg.portfolio.bellman"   , level = "DETAIL")
    get_logger("borg.portfolio._bellman"  , level = "DETAIL")
    get_logger("borg.portfolio.sat_world" , level = "DETAIL")
    get_logger("borg.portfolio.strategies", level = "DETAIL")

    # build the trainer, if any
    if url:
        from cargo.sql.alchemy    import (
            make_engine,
            make_session,
            )
        from borg.portfolio.world import DecisionTrainer

        trainer = DecisionTrainer(make_session(bind = make_engine(url)), train_uuids)
    else:
        trainer = None

    # construct the solver
    def built(solver):
        """
        Receive a solver from the construction script.
        """

        import cPickle as pickle

        with open(out_path, "w") as file:
            pickle.dump(solver, file, -1)

    execfile(solver_py, {"built" : built, "trainer" : trainer})

