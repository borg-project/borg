"""
utexas/sat/solvers/preprocessing.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log               import get_logger
from utexas.sat.solvers.base import (
    SAT_Result,
    SAT_Solver,
    )

log = get_logger(__name__)

class SAT_PreprocessingSolverResult(SAT_Result):
    """
    Outcome of a solver with a preprocessing step.
    """

    def __init__(self, preprocessor_output, solver_result, certificate):
        """
        Initialize.
        """

        SAT_Result.__init__(self)

        self.preprocessor_output = preprocessor_output
        self.solver_result       = solver_result
        self._certificate        = certificate

    @property
    def satisfiable(self):
        """
        Did the solver report the instance satisfiable?
        """

        if self.solver_result is None:
            return self.preprocessor_output.solver_result.satisfiable
        else:
            return self.solver_result.satisfiable

    @property
    def certificate(self):
        """
        Certificate of satisfiability, if any.
        """

        return self._certificate

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
        self.solver       = solver

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        from cargo.io       import mkdtemp_scoped
        from cargo.temporal import TimeDelta

        # FIXME improve no-cutoff support
        if cutoff is None:
            cutoff = TimeDelta(seconds = 1e6)

        with mkdtemp_scoped(prefix = "sat_preprocessing.") as sandbox_path:
            preprocessed = self.preprocessor.preprocess(input_path, sandbox_path, cutoff)

            if preprocessed.solver_result is not None:
                # the preprocessor solved the instance
                return \
                    SAT_PreprocessingSolverResult(
                        preprocessed,
                        None,
                        preprocessed.solver_result.certificate,
                        )
            else:
                # the preprocessor did not solve the instance
                remaining = max(TimeDelta(), cutoff - preprocessed.elapsed)

                if preprocessed.cnf_path is None:
                    # ... it failed unexpectedly
                    result   = self.solver.solve(input_path, remaining, seed)
                    extended = result.certificate
                else:
                    # ... it generated a new CNF
                    result = self.solver.solve(preprocessed.cnf_path, remaining, seed)

                    if result.certificate is None:
                        extended = None
                    else:
                        extended = preprocessed.extend(result.certificate)

                return SAT_PreprocessingSolverResult(preprocessed, result, extended)

