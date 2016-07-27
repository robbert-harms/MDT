import numpy as np
from mdt.utils import ChunksProcessingStrategy

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'Fit in chunks of voxel ranges',
             'description': 'Processes a model in chunks defined by ranges of voxels.'}


class VoxelRange(ChunksProcessingStrategy):

    def __init__(self, nmr_voxels=40000, **kwargs):
        """Optimize a given dataset in batches of the given number of voxels

        Args:
            nmr_voxels (int): the number of voxels per batch

        Attributes:
            nmr_voxels (int): the number of voxels per chunk
        """
        super(VoxelRange, self).__init__(**kwargs)
        self.nmr_voxels = nmr_voxels

    def _chunks_generator(self, model, problem_data, output_path, worker):
        if self._honor_voxels_to_analyze and model.problems_to_analyze:
            self._logger.info('The range of problems to analyze was already set, '
                              'we will only fit the selected problems.')
            roi_indices = model.problems_to_analyze
        else:
            roi_indices = np.arange(0, np.count_nonzero(problem_data.mask))

        for ind_start in range(0, len(roi_indices), self.nmr_voxels):
            ind_end = min(len(roi_indices), ind_start + self.nmr_voxels)
            yield roi_indices[ind_start:ind_end]
