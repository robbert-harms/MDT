from mdt.components_loader import ProcessingStrategiesLoader
from mdt.utils import ModelChunksProcessingStrategy

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'All slices at once',
             'description': 'Processes the whole dataset at once. No intermediate results are saved.'}


class AllVoxelsAtOnce(ModelChunksProcessingStrategy):
    """Run all slices at once."""

    def run(self, model, problem_data, output_path, recalculate, worker):
        if self.honor_voxels_to_analyze and model.problems_to_analyze:
            self._logger.info('The range of problems to analyze was already set, '
                              'we will only fit the selected problems.')
            strategy = ProcessingStrategiesLoader().load('VoxelRange', honor_voxels_to_analyze=True)
            return strategy.run(model, problem_data, output_path, recalculate, worker)
        else:
            self._logger.info('Processing all voxels at once')
            return worker.process(model, problem_data, problem_data.mask, output_path, output_path)
