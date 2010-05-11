"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

class SingleEvaluation(object):
    """
    """

class Evaluation(object):
    """
    Evaluate the performance of solvers.
    """

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

