from cargo.temporal import *
from borg.solvers   import *
from borg.analyzers import *
from borg.portfolio import *

actions = \
    map(
        lambda s: SolverAction(LookupSolver(s), seconds(1.0)),
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

