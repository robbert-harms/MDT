from mdt.components_config.cascade_models import CascadeConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SSFP_Tensor_ExVivo(CascadeConfig):

    name = 'SSFP_Tensor-ExVivo (Cascade)'
    description = 'Cascade for SSFP Tensor with ex vivo defaults.'
    models = ('SSFP_BallStick_r1-ExVivo (Cascade)',
              'SSFP_Tensor-ExVivo')
