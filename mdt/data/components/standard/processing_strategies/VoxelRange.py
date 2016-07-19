import os
import numpy as np
import shutil
from mdt.utils import ModelChunksProcessingStrategy, create_roi, restore_volumes

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'Fit in chunks of voxel ranges',
             'description': 'Processes a model in chunks defined by ranges of voxels.'}


class VoxelRange(ModelChunksProcessingStrategy):

    def __init__(self, nmr_voxels=40000, honor_voxels_to_analyze=True):
        """Optimize a given dataset in batches of the given number of voxels

        Args:
            nmr_voxels (int): the number of voxels per batch
            honor_voxels_to_analyze (bool): if set to True, we use the model's voxels_to_analyze setting if it is set
                instead of fitting all voxels in the mask

        Attributes:
            nmr_voxels (int): the number of voxels per chunk
            honor_voxels_to_analyze (bool): if set to True, we use the model's voxels_to_analyze setting if it is set
                instead of fitting all voxels in the mask
        """
        super(VoxelRange, self).__init__(honor_voxels_to_analyze=honor_voxels_to_analyze)
        self.nmr_voxels = nmr_voxels

    def run(self, model, problem_data, output_path, recalculate, worker):
        tmp_storage_dir = os.path.join(output_path, self.tmp_results_subdir)
        mask = problem_data.mask
        mask_list = create_roi(mask, mask)

        if self.honor_voxels_to_analyze and model.problems_to_analyze:
            self._logger.info('The range of problems to analyze was already set, '
                              'we will only fit the selected problems.')
            indices = model.problems_to_analyze
        else:
            indices = np.arange(0, np.count_nonzero(mask))

        self._prepare_tmp_storage_dir(tmp_storage_dir, recalculate)

        for ind_start, ind_end in self._chunks_generator(len(indices)):
            chunk_indices = indices[ind_start:ind_end]

            mask_list[:] = 0
            mask_list[ind_start:ind_end] = 1
            chunk_mask = restore_volumes(mask_list, mask, with_volume_dim=False)

            if len(chunk_indices):
                with self._selected_indices(model, chunk_indices):
                    self._run_on_chunk(model, problem_data, tmp_storage_dir, worker, ind_start, ind_end, chunk_mask)

        self._logger.info('Computed all slices, now merging the results')
        return_data = worker.combine(model, problem_data, tmp_storage_dir, output_path)
        shutil.rmtree(tmp_storage_dir)
        return return_data

    def _chunks_generator(self, total_nmr_voxels):
        """Generate the slices/chunks we will use for the fitting.

        Args:
            total_nmr_voxels (int): the total number of voxels to fit

        Returns:
            tuple (int, int, list): the start of the slice index, the end of the slice index and the list with
                the slices to select from the mask.
        """
        for ind_start in range(0, total_nmr_voxels, self.nmr_voxels):
            ind_end = min(total_nmr_voxels, ind_start + self.nmr_voxels)
            yield ind_start, ind_end

    def _run_on_chunk(self, model, problem_data, tmp_storage_dir, worker, ind_start, ind_end, tmp_mask):
        if worker.output_exists(model, tmp_mask, tmp_storage_dir):
            self._logger.info('Skipping voxels {} to {}, they are already processed.'.format(ind_start, ind_end))
        else:
            self._logger.info('Computing voxels {0} up to {1} ({2} voxels in total, currently {3:.2%} is processed)'.
                              format(ind_start, ind_end, np.count_nonzero(problem_data.mask),
                                     float(ind_start) / np.count_nonzero(problem_data.mask)))

            worker.process(model, problem_data, tmp_mask, tmp_storage_dir)
