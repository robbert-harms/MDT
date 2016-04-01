import tempfile
import shutil
import numpy as np
from mdt import fit_model, restore_volumes, create_roi, create_median_otsu_brain_mask, config_context, \
    yaml_string_to_dict
from mdt.utils import NoiseStdEstimator, load_problem_data, MetaOptimizerBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-11-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SubtractEstimatedS0(NoiseStdEstimator):

    def calculate(self, **kwargs):
        """Calculate the standard deviation of the error using the unweighted volumes.

        We first fit a S0 model to the data, and subtract this estimate from the unweighted volumes. Next, we
        compute per voxel the E[S^2] and use sqrt(E[S^2]/2) as estimate for the noise std in that voxel.

        Finally, we calculate the mean of all those noise std estimates.

        Returns:
            float: single value representing the sigma for the given volume
        """
        if self._mask is None:
            self._mask = create_median_otsu_brain_mask(self._signal4d, self._protocol)

        s0_vol = self._get_s0_fit()
        unweighted_indices = self._protocol.get_unweighted_indices()
        unweighted_volumes = self._signal4d[..., unweighted_indices]
        baseline_images = unweighted_volumes - s0_vol
        voxel_values = create_roi(baseline_images, self._mask)

        sigmas = np.sqrt(np.mean(np.power(voxel_values, 2), axis=1) / 2)
        return np.mean(sigmas)

    def _get_s0_fit(self):
        self._logger.info('Estimating S0 for the noise standard deviation')
        tmp_dir = tempfile.mkdtemp()

        config = '''
        optimization_settings:
            general:
                optimizers:
                    -   name: 'Powell'
                        patience: 2
        '''
        with config_context(yaml_string_to_dict(config)):
            output = fit_model('S0',
                               load_problem_data([self._signal4d, None], self._protocol, self._mask),
                               tmp_dir,
                               noise_std=None)
        shutil.rmtree(tmp_dir)
        self._logger.info('Done fitting S0 for the noise standard deviation')
        return restore_volumes(output['S0.s0'], self._mask)
