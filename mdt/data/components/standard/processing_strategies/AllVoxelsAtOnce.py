import numpy as np

from mdt.components_loader import ProcessingStrategiesLoader
from mdt.utils import SimpleProcessingStrategy

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'All slices at once',
             'description': 'Processes the whole dataset at once.'}


class AllVoxelsAtOnce(SimpleProcessingStrategy):
    """Run all slices at once."""

    def run(self, model, problem_data, output_path, recalculate, worker):
        if self._honor_voxels_to_analyze and model.problems_to_analyze:
            self._logger.info('The range of problems to analyze was already set, '
                              'we will only fit the selected problems.')

            strategy = ProcessingStrategiesLoader().load('VoxelRange', honor_voxels_to_analyze=True,
                                                         tmp_dir=self._tmp_dir)

            return strategy.run(model, problem_data, output_path, recalculate, worker)
        else:
            self._logger.info('Processing all voxels at once')

            with self._tmp_storage_dir(output_path, recalculate) as tmp_storage_dir:
                worker.process(model, problem_data, np.arange(0, np.count_nonzero(problem_data.mask)), tmp_storage_dir)
                return_data = worker.combine(model, problem_data, tmp_storage_dir, output_path)

            return return_data
