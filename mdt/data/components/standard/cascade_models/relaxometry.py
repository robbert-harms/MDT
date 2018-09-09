from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-08-26'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class S0T1_MI_EPI(CascadeTemplate):

    models = ('S0',
              ('S0T1_MI_EPI', 'S0T1_MI_EPI_fixed_eff'),
              'S0T1_MI_EPI')
    fixes = {'S0T1_MI_EPI_fixed_eff': {'ExpT1DecIR.Efficiency': 1}}


class S0_T2(CascadeTemplate):

    models = ('S0',
              'S0-T2')
