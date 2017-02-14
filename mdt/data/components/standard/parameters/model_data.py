"""Definitions of the model data parameters.

These are meant for model specific data that the model needs to work. You can of course inline these variables in
the code for one of the models (which is faster), but this way lets the user change the specifics of the
model by changing the data in the model data parameters.

Please choose the parameter type for a model and parameter carefully since the type signifies how the parameter and
its data are handled during model construction.

"""

import numpy as np
from mdt.components_config.parameters import ModelDataParameterConfig

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


# charmed default, used in GDRCylindersFixed model
class gamma_radii(ModelDataParameterConfig):

    data_type = 'global const mot_float_type* const'
    value = 1e-6 * np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])


class gamma_cyl_weights(ModelDataParameterConfig):

    data_type = 'global const mot_float_type* const'
    value = np.array([0.0211847200855742, 0.107169623942214,
                      0.194400551313197, 0.266676876170322,
                      0.214921653661151, 0.195646574827541])


class nmr_gamma_cyl_weights(ModelDataParameterConfig):

    data_type = 'int'
    value = 6
