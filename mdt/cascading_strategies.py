__author__ = 'Robbert Harms'
__date__ = "2016-09-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class OptimizationCascadingStrategy(object):

    def __init__(self):
        """A cascading strategy defines how a single model is optimized using information from simpler models."""

    def process(self, model, problem_data):
        """Process the given model and problem data using this cascading strategy.

        Args:
            model (DMRISingleModel): the model we want to process in the end
            problem_data (DMRIProblemData): the problem data to use for all processing
        """


class SamplingCascadingStrategy(object):

    def __init__(self):
        """A cascading strategy defines how a single model is sampled using information from simpler models."""

    def process(self, model, problem_data, sampler):
        """Process the given model and problem data using this cascading strategy.

        Args:
            model (DMRISingleModel): the model we want to process in the end
            problem_data (DMRIProblemData): the problem data to use for all processing
            sampler (Sampler): the sampler to use
        """


class DirectSamplingStrategy(SamplingCascadingStrategy):

    def process(self, model, problem_data, sampler):
        pass
