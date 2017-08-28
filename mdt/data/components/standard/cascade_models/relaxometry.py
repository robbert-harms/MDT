from mdt.component_templates.cascade_models import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-08-26'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class S0_T1_GRE(CascadeTemplate):

    description = 'Cascade for S0_T1_GRE model.'
    models = ('S0',
              'S0_T1_GRE',
              'S0_T1_GRE')
    fixes = {1: [('ExpT1DecIR.Efficiency', 1)]}
