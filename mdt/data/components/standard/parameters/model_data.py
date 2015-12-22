import numpy as np

from mdt.models.parameters import ModelDataParameterConfig
from mot.base import DataType

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


# charmed default, used in GDRCylindersFixed model
class gamma_radii(ModelDataParameterConfig):

    name = 'gamma_radii'
    data_type = DataType.from_string('MOT_FLOAT_TYPE*').set_address_space_qualifier('global')\
                                                       .set_pre_data_type_type_qualifiers(['const'])\
                                                       .set_post_data_type_type_qualifier('const')
    value = 1e-6 * np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])


class gamma_cyl_weights(ModelDataParameterConfig):

    name = 'gamma_cyl_weights'
    data_type = DataType.from_string('MOT_FLOAT_TYPE*').set_address_space_qualifier('global')\
                                                       .set_pre_data_type_type_qualifiers(['const'])\
                                                       .set_post_data_type_type_qualifier('const')
    value = np.array([0.0211847200855742, 0.107169623942214,
                      0.194400551313197, 0.266676876170322,
                      0.214921653661151, 0.195646574827541])


class nmr_gamma_cyl_weights(ModelDataParameterConfig):

    name = 'nmr_gamma_cyl_weights'
    data_type = DataType.from_string('int')
    value = 6
