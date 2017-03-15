from mdt.components_loader import load_component
from mdt.processing_strategies import SimpleProcessingStrategy

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'Applies the VoxelRange strategy depending on the protocol.',
             'description': 'This looks at the size of the protocol and based on that determines the voxel range.'}


class ProtocolDependent(SimpleProcessingStrategy):

    def __init__(self, steps=((0, None), (100, 50000), (200, 20000)), **kwargs):
        """A meta strategy using VoxelRange or AllVoxelsAtOnce depending on the protocol length

        This will look at the protocol of the given model and determine, based on the number of rows in the protocol,
        which voxel range to use.

        A voxel range of None or 0 means we want to fit all the voxels at once (i.e. we use the
        AllVoxelsAtOnce strategy).

        During lookup of the protocol length we take the maximum step that is lower than the protocol length. If no
        suitable lookup is present, we use the AllVoxelsAtOnce strategy.

        Args:
            steps (list[tuple[int, int]]): the steps of the voxel ranges. The first item in the tuple is the
                protocol length, the second the voxel range. We assume that voxel ranges are in ascending order.
        """
        super(ProtocolDependent, self).__init__(**kwargs)
        self._steps = steps
        self._kwargs = kwargs

    def run(self, model, problem_data, output_path, recalculate, worker_generator):
        strategy = self._get_strategy(problem_data)
        return strategy.run(model, problem_data, output_path, recalculate, worker_generator)

    def _get_strategy(self, problem_data):
        for col_length, voxel_range in reversed(self._steps):
            if int(col_length) < problem_data.get_nmr_inst_per_problem():
                if voxel_range:
                    return load_component('processing_strategies', 'VoxelRange', nmr_voxels=int(voxel_range),
                                          **self._kwargs)

        return load_component('processing_strategies', 'AllVoxelsAtOnce', **self._kwargs)
