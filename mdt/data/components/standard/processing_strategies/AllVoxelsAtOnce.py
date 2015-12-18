from mdt.utils import ModelProcessingStrategy

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'All slices at once',
             'description': 'Processes the whole dataset at once. No intermediate results are saved.'}


class AllVoxelsAtOnce(ModelProcessingStrategy):
    """Run all slices at once."""

    def run(self, model, problem_data, output_path, recalculate, worker):
        self._logger.info('Processing all voxels at once')
        return worker.process(model, problem_data, problem_data.mask, output_path)
