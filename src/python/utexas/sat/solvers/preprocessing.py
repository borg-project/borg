"""
utexas/sat/solvers/preprocessing.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log               import get_logger
from cargo.temporal          import TimeDelta
from utexas.sat.solvers.base import (
    SAT_Solver,
    SAT_BareResult,
    )

log = get_logger(__name__)

class SAT_PreprocessingSolverResult(SAT_BareResult):
    """
    Outcome of a solver with a preprocessing step.
    """

    def __init__(self, solver, task, preprocessor_output, solver_result, certificate):
        """
        Initialize.
        """

        if self.solver_result is None:
            satisfiable = self.preprocessor_output.solver_result.satisfiable
        else:
            satisfiable = self.solver_result.satisfiable

        SAT_BareResult.__init__(
            self,
            solver,
            task,
            run.cutoff,
            run.proc_elapsed,
            satisfiable,
            certificate,
            )

        self.preprocessor_output = preprocessor_output
        self.solver_result       = solver_result

    def to_orm(self):
        """
        Return a database description of this result.
        """

        if self.solver_result is None:
            inner_attempt_row = None
        else:
            inner_attempt_row = self.solver_result.to_orm()

        attempt_row       = \
            SAT_PreprocessingAttemptRow(
                run           = CPU_LimitedRunRow.from_run(self.preprocessor_output.run),
                preprocessor  = self.solver.preprocessor.to_orm(),
                inner_attempt = inner_attempt_row,
                preprocessed  = self.preprocessor_output.preprocessed,
                )

        return self.update_orm(attempt_row)

class SAT_PreprocessingSolver(SAT_Solver):
    """
    Execute a solver after a preprocessor pass.
    """

    def __init__(self, preprocessor, solver):
        """
        Initialize.
        """

        SAT_Solver.__init__(self)

        self.preprocessor = preprocessor
        self.inner_solver = solver

    def solve(self, task, cutoff = TimeDelta(seconds = 1e6), seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        from cargo.io import mkdtemp_scoped

        with mkdtemp_scoped(prefix = "sat_preprocessing.") as sandbox_path:
            preprocessed = self.preprocessor.preprocess(task, sandbox_path, cutoff)

            if preprocessed.solver_result is not None:
                # the preprocessor solved the instance
                return \
                    SAT_PreprocessingSolverResult(
                        self,
                        task,
                        preprocessed,
                        None,
                        preprocessed.solver_result.certificate,
                        )
            else:
                # the preprocessor did not solve the instance
                remaining = max(TimeDelta(), cutoff - preprocessed.elapsed)

                if preprocessed.cnf_path is None:
                    # ... it failed unexpectedly
                    result   = self.inner_solver.solve(task, remaining, seed)
                    extended = result.certificate
                else:
                    # ... it generated a new CNF
                    result = self.inner_solver.solve(preprocessed.cnf_path, remaining, seed)

                    if result.certificate is None:
                        extended = None
                    else:
                        extended = preprocessed.extend(result.certificate)

                return \
                    SAT_PreprocessingSolverResult(
                        self,
                        task,
                        preprocessed,
                        result,
                        extended,
                        )

