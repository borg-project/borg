"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from utexas.tools.evaluate import main

    raise SystemExit(main())

from cargo.flags        import (
    Flag,
    Flags,
    )

log          = get_logger(__name__, level = "NOTE")
module_flags = \
    Flags(
        "Script Options",
        Flag(
            "-t",
            "--training",
            default = "training.json",
            metavar = "FILE",
            help    = "load training data from FILE [%default]",
            ),
        Flag(
            "-o",
            "--output",
            default = "portfolio.json",
            metavar = "FILE",
            help    = "write configuration to FILE [%default]",
            ),
        )

def get_solvers():
    """
    Build all solvers.
    """

def get_tasks():
    """
    Build all tasks.
    """

class Evaluation(object):
    def __init__(self, budget, engine, random, label):
        """
        Initialize.
        """

        # members
        from cargo.sql.alchemy import session_maker

        self.budget  = budget
        self.Session = session_maker(bind = engine)
        self.random  = random

        # create a trial for this evaluation
        with self.Session() as session:
            self.trial_row = SAT_TrialRow(label = label)

            session.add(self.trial_row)
            session.commit()

    def make_attempts(solver, task_pairs):
        """
        Run the solver on the specified tasks.
        """

        with self.Session() as session:
            for (task, task_row) in task_pairs:
                result  = solver.solve(task, self.budget, seed = self.random)
                attempt = \
                    SAT_AttemptRow(
                        budget      = self.budget,
                        cost        = result.cost,
                        satisfiable = result.satisfiable,
                        certificate = result.certificate,
                        task        = task_row,
                        trials      = [trial_row],
                        )

                session.add(attempt)

            session.commit()

    def evaluate(solvers, tasks):
        """
        Run solvers on the specified tasks.
        """

        for solver in solvers:
            self.make_attempts(solver, tasks)

def main():
    """
    Application body.
    """

    # get command line arguments
    import utexas.sat.solvers

    from cargo.flags import parse_given

    parse_given()

    # set up log output
    from cargo.log import enable_default_logging

    enable_default_logging()

    # build the solvers

