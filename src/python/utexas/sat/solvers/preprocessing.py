"""
utexas/sat/solvers/preprocessing.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.solvers.base import SAT_Solver

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

    @abstractproperty
    def satisfiable(self):
        """
        Did the solver report the instance satisfiable?
        """

        return self.solver_result.satisfiable

    @abstractproperty
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
            remaining    = max(TimeDelta(), cutoff - preprocessed.elapsed)
            result       = self.solver.solve(preprocessed.cnf_path, remaining, seed)

            if result.certificate is None:
                extended = None
            else:
                extended = preprocessed.extend(result.certificate)

            return SAT_PreprocessingSolverResult(preprocessed, result, extended)

