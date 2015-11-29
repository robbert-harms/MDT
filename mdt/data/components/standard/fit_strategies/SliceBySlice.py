import os
import numpy as np
import shutil
from contextlib import contextmanager
from mdt.utils import model_output_exists, create_roi, ModelChunksFitting

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
        slices_dir = os.path.join(output_path, 'slices')
        self._prepare_chunk_dir(slices_dir, recalculate)

        for ind_start, ind_end, slicer in self._slicing_generator(mask):
            if mask[slicer].any():
                with self._mask_context(problem_data, tmp_mask, ind_start, ind_end) as slice_problem_data:
                    self._run_on_slice(model, slice_problem_data, slices_dir, optimizer, recalculate,
                                       ind_start, ind_end)

        return self._join_chunks(output_path, slices_dir)

    def _slicing_generator(self, mask):
        slicer = [slice(None)] * len(mask.shape)
        dimension_length = mask.shape[self.slice_dimension]

        for ind_start in range(0, dimension_length, self.slice_width):
            ind_end = min(dimension_length, ind_start + self.slice_width)
            slicer[self.slice_dimension] = slice(ind_start, ind_end)

            yield ind_start, ind_end, slicer

    def _run_on_slice(self, model, problem_data, slices_dir, optimizer, recalculate, ind_start, ind_end):
        slice_dir = os.path.join(slices_dir, '{dimension}_{start}_{end}'.format(
            dimension=self.slice_dimension, start=ind_start, end=ind_end))

        model.set_problem_data(problem_data)

        if recalculate and os.path.exists(slice_dir):
            shutil.rmtree(slice_dir)

        if model_output_exists(model, slice_dir, append_model_name_to_path=False):
            self._logger.info('Skipping slices {} to {}, they already exist.'.format(ind_start, ind_end))
        else:
            self._logger.info('Computing slices {} to {}'.format(ind_start, ind_end))
            results, extra_output = optimizer.minimize(model, full_output=True)
            results.update(extra_output)
            results.update({'__mask': create_roi(problem_data.mask, problem_data.mask)})
            self._write_output(results, problem_data.mask, slice_dir, problem_data.volume_header)

    @contextmanager
    def _mask_context(self, problem_data, tmp_mask, slice_ind_start, slice_ind_end):
        """Create a context in which the mask in the problem data is set to a single slice mask.

        Args:
            problem_data: the problem data to set with the new mask
            tmp_mask: the temporary mask array to use. We could have created the tmp mask in this routine,
                but that would mean we have to do it for every new slice. Faster is to give it and this routine will
                set it to 0.
            slice_ind_start, the start of the slice
            slice_ind_end: the end of the slice
        """
        old_mask = problem_data.mask

        tmp_mask.fill(0)
        slicing = [slice(None)] * len(old_mask.shape)
        slicing[self.slice_dimension] = slice(slice_ind_start, slice_ind_end)
        tmp_mask[slicing] = old_mask[slicing]

        problem_data.mask = tmp_mask
        yield problem_data
        problem_data.mask = old_mask
