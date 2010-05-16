"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.preprocessors import SAT_Preprocessor

class LookupPreprocessor(SAT_Preprocessor):
    """
    Use a named preprocessor.
    """

    def __init__(self, name):
        """
        Initialize.
        """

        SAT_Preprocessor.__init__(self)

        self._name = name

    def preprocess(self, task, budget, output_dir, random, environment):
        """
        Preprocess an instance.
        """

        from utexas.sat.preprocessors import WrappedPreprocessorResult

        inner        = environment.named_preprocessors[self._name]
        inner_result = inner.preprocess(task, budget, output_dir, random, environment)

        return WrappedPreprocessorResult(self, inner_result)

