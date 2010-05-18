"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.preprocessors import SAT_Preprocessor
from utexas.rowed             import Rowed

class LookupPreprocessor(Rowed, SAT_Preprocessor):
    """
    Use a named preprocessor.
    """

    def __init__(self, name):
        """
        Initialize.
        """

        Rowed.__init__(self)

        self._name = name

    def preprocess(self, task, budget, output_path, random, environment):
        """
        Preprocess an instance.
        """

        from utexas.sat.preprocessors import WrappedPreprocessorResult

        inner        = self.look_up(environment)
        inner_result = inner.preprocess(task, budget, output_path, random, environment)

        return WrappedPreprocessorResult(self, inner_result)

    def extend(self, task, answer, environment):
        """
        Pretend to extend an answer.
        """

        return self.look_up(environment).extend(task, answer, environment)

    def look_up(self, environment):
        """
        Look up the named preprocessor.
        """

        return environment.named_preprocessors[self._name]

    def get_new_row(self, session):
        """
        Create or obtain an ORM row for this object.
        """

        from utexas.data import PreprocessorRow

        return session.query(PreprocessorRow).get(self._name)

