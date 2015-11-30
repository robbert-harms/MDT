import os
import numpy as np
import shutil
from contextlib import contextmanager
from mdt.utils import model_output_exists, create_roi, ModelChunksFitting, restore_volumes

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'Slice by slice fitting',
             'description': 'Fitting a model slice by slice, saving the intermediate results.'}


class SliceBySlice(ModelChunksFitting):

    def __init__(self, slice_dimension=2, slice_width=1):
        """Optimize a given dataset slice by slice.

        Args:
            slice_dimension: in which dimension to slice the dataset
            slice_width: the width of the slices in that dimension

        Attributes:
            slice_dimension: in which dimension to slice the dataset
            slice_width: the width of the slices in that dimension
        """
        super(SliceBySlice, self).__init__()
        self.slice_dimension = slice_dimension
        self.slice_width = slice_width

    def run(self, model, problem_data, output_path, optimizer, recalculate):
        """Optimize slice by slice"""
        mask = problem_data.mask
        tmp_mask = np.zeros_like(mask)
        indices = self._get_index_matrix(problem_data.mask)
        slices_dir = os.path.join(output_path, 'slices')
        self._prepare_chunk_dir(slices_dir, recalculate)

        for ind_start, ind_end, slicer in self._slicing_generator(mask):
            tmp_mask.fill(0)
            tmp_mask[slicer] = mask[slicer]

            if tmp_mask.any():
                with self._selected_indices(model, indices, mask, slicer):
                    self._run_on_slice(model, problem_data, slices_dir, optimizer, recalculate,
                                       ind_start, ind_end, tmp_mask)

        return self._join_chunks(output_path, slices_dir)

    def _slicing_generator(self, mask):
        slicer = [slice(None)] * len(mask.shape)
        dimension_length = mask.shape[self.slice_dimension]

        for ind_start in range(0, dimension_length, self.slice_width):
            ind_end = min(dimension_length, ind_start + self.slice_width)
            slicer[self.slice_dimension] = slice(ind_start, ind_end)

            yield ind_start, ind_end, slicer

    def _run_on_slice(self, model, problem_data, slices_dir, optimizer, recalculate, ind_start, ind_end, tmp_mask):
        slice_dir = os.path.join(slices_dir, '{dimension}_{start}_{end}'.format(
            dimension=self.slice_dimension, start=ind_start, end=ind_end))

        if recalculate and os.path.exists(slice_dir):
            shutil.rmtree(slice_dir)

        if model_output_exists(model, slice_dir, append_model_name_to_path=False):
            self._logger.info('Skipping slices {} to {}, they already exist.'.format(ind_start, ind_end))
        else:
            self._logger.info('Computing slices {} to {}'.format(ind_start, ind_end))
            results, extra_output = optimizer.minimize(model, full_output=True)
            results.update(extra_output)
            results.update({'__mask': create_roi(tmp_mask, tmp_mask)})
            self._write_output(results, tmp_mask, slice_dir, problem_data.volume_header)

    @contextmanager
    def _selected_indices(self, model, indices, mask, slicer):
        """Create a context in which problems_to_analyze attribute of the models is set to the selected indices.

        Args:
            model: the model to which to set the problems_to_analyze
            indices (ndarray): the volume with the indices
            mask (ndarray): the mask with the selected voxels
            slicer (list of slices): the slices we want to select
        """
        old_setting = model.problems_to_analyze

        masked = np.nonzero(mask[slicer].flatten())[0]
        selected_indices = indices[slicer].flatten()[masked]

        model.problems_to_analyze = selected_indices
        yield
        model.problems_to_analyze = old_setting
