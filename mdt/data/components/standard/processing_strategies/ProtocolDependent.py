import os
import numpy as np
import shutil

from mdt import create_index_matrix
from mdt.IO import Nifti
from mdt.components_loader import ProcessingStrategiesLoader, load_component
from mdt.utils import ModelChunksProcessingStrategy

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'Applies the VoxelRange strategy depending on the protocol.',
             'description': 'This looks at the size of the protocol and based on that determines the voxel range.'}


class ProtocolDependent(ModelChunksProcessingStrategy):

    def __init__(self, steps=((0, None), (100, 50000), (200, 20000))):
        """Optimize a given dataset slice by slice.

        This will look at the protocol of the given model and determine based on the number of rows in the protocol
        which voxel range to use.

        A voxel range of None or 0 means we want to fit all the voxels at once (We will use the AllVoxelsAtOnce
        strategy for that).

        During lookup of the protocol length we take the maximum step that is lower than the protocol length. If no
        suitable lookup is present, we use the AllVoxelsAtOnce strategy.

        Args:
            steps (list[tuple[int, int]]): the steps of the voxel ranges. The first item in the tuple is the
                protocol length, the second the voxel range. We assume the voxel ranges to be in ascending order.
        """
        super(ProtocolDependent, self).__init__()
        self._steps = steps

    def run(self, model, problem_data, output_path, recalculate, worker):
        strategy = self._get_strategy(problem_data)
        return strategy.run(model, problem_data, output_path, recalculate, worker)

    def _get_strategy(self, problem_data):
        for col_length, voxel_range in reversed(self._steps):
            if int(col_length) < problem_data.protocol.length:
                if voxel_range:
                    return load_component('processing_strategies', 'VoxelRange', nmr_voxels=int(voxel_range))

        return load_component('processing_strategies', 'AllVoxelsAtOnce')
