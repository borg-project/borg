from datetime       import timedelta
from borg.solvers   import *
from borg.analyzers import *
from borg.portfolio import *

actions = \
    map(
        lambda s: SolverAction(LookupSolver(s), timedelta(seconds = 1.0)),
        ["foo", "bar", "baz"],
        )

built(
    PortfolioSolver(
        ModelingStrategy(
            RandomModel(actions),
            HardMyopicPlanner(1.0),
            ),
        NoAnalyzer(),
        ),
    )

