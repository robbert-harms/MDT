from mdt.utils import ModelFitStrategy

__author__ = 'Robbert Harms'
__date__ = "2015-11-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'title': 'All slices at once',
             'description': 'Fits a model to the whole dataset as once. No intermediate results are saved.'}


class AllSlicesAtOnce(ModelFitStrategy):
    """Run all slices at once."""

    def run(self, model, problem_data, output_path, optimizer, recalculate):
        self._logger.info('Fitting all slices at once')
        results, extra_output = optimizer.minimize(model, full_output=True)
        results.update(extra_output)
        self._write_output(results, problem_data.mask, output_path, problem_data.volume_header)
        return results
