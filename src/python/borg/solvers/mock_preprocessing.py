"""
utexas/sat/solvers/preprocessing.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log    import get_logger
from borg.rowed   import Rowed
from borg.solvers import (
    AbstractSolver,
    PreprocessingAttempt,
    )

log = get_logger(__name__)

class RecyclingPreprocessingSolver(Rowed, AbstractSolver):
    """
    Execute a solver after a preprocessor pass.
    """

    def __init__(self, preprocessor_name, solver):
        """
        Initialize.
        """

        SAT_Solver.__init__(self)

        self.preprocessor_name = preprocessor_name
        self.inner_solver      = solver

    def solve(self, task, budget, random, environment):
        """
        Execute the solver and return its outcome, given an input task.
        """

        # argument sanity
        from utexas.sat.tasks import MockTask

        if not isinstance(task, MockTask):
            raise TypeError("mock solver requires a mock task")

        # mise en place
        from sqlalchemy  import (
            and_,
            )
        from utexas.data import (
            SAT_TrialRow       as ST,
            SAT_AttemptRow     as SA,
            PreprocessorRunRow as PRR,
            )

        # generate a recycled result
        with environment.CacheSession() as session:
            # select a preprocessor run
            from sqlalchemy.sql.functions import random as sql_random

            run_row = \
                session \
                .query(PRR) \
                .filter(
                    and_(
                        PRR.preprocessor_name == self.preprocessor_name,
                        PRR.input_task_uuid   == task.task_uuid,
                        PRR.budget            >= budget,
                        PRR.trials.contains(ST.get_recyclable(session)),
                        ),
                    ) \
                .order_by(sql_random()) \
                .first()

            if run_row is None:
                raise RuntimeError("database does not contain a matching recyclable run")

            # interpret the run
            if run_row.cost > budget:
                return SAT_BareResult(self, task, budget, budget, None, None)
            elif run_row.answer is not None:
                return \
                    SAT_BareResult(
                        self,
                        task,
                        budget,
                        run_row.cost,
                        run_row.answer.satisfiable,
                        run_row.answer.get_certificate(),
                        )
            else:
                preprocessor_run_cost         = run_row.cost
                preprocessor_output_task_uuid = run_row.output_task_uuid

        # not solved by the preprocessor; try the inner solver
        inner_result = \
            self.inner_solver.solve(
                MockTask(preprocessor_output_task_uuid),
                budget - preprocessor_run_cost,
                random,
                environment,
                )

        return \
            SAT_BareResult(
                self,
                task,
                budget,
                inner_result.cost + preprocessor_run_cost,
                inner_result.satisfiable,
                None,
                )

