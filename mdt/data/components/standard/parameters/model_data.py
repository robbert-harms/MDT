import numpy as np
from mdt.models.parameters import ModelDataParameterConfig

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


# charmed default, used in GDRCylindersFixed model
class gamma_radii(ModelDataParameterConfig):

    name = 'gamma_radii'
    data_type = 'global const MOT_FLOAT_TYPE* const'
    value = 1e-6 * np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])


class gamma_cyl_weights(ModelDataParameterConfig):

    name = 'gamma_cyl_weights'
    data_type = 'global const MOT_FLOAT_TYPE* const'
    value = np.array([0.0211847200855742, 0.107169623942214,
                      0.194400551313197, 0.266676876170322,
                      0.214921653661151, 0.195646574827541])


class nmr_gamma_cyl_weights(ModelDataParameterConfig):

    name = 'nmr_gamma_cyl_weights'
    data_type = 'int'
    value = 6
