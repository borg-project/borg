"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.preprocessors import SAT_Preprocessor

class FixedPreprocessor(SAT_Preprocessor):
    """
    A fake, fixed-result preprocessor.
    """

    def __init__(self, output_task, answer):
        """
        Initialize.
        """

        self._output_task = output_task
        self._answer      = answer

    def preprocess(self, task, budget, output_dir, random, environment):
        """
        Pretend to preprocess an instance.
        """

        from utexas.sat.preprocessors import BarePreprocessorResult

        return \
            BarePreprocessorResult(
                self,
                task,
                self._output_task,
                budget,
                budget,
                self._answer,
                )

    def extend(self, task, answer):
        """
        Pretend to extend an answer.
        """

        raise NotImplementedError()

