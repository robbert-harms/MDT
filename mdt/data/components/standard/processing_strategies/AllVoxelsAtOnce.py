from mdt.processing_strategies import ChunksProcessingStrategy

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'All slices at once',
             'description': 'Processes the whole dataset at once.'}


class AllVoxelsAtOnce(ChunksProcessingStrategy):
    """Run all slices at once."""

    def _chunks_generator(self, model, problem_data, output_path, worker, total_roi_indices):
        yield total_roi_indices
