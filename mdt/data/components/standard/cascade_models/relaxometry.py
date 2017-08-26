from mdt.component_templates.cascade_models import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-08-26'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class S0TIGre(CascadeTemplate):

    description = 'Cascade for S0-TIGre model.'
    models = ('S0',
              'S0-TIGre',
              'S0-TIGre')
    fixes = {1: [('ExpT1DecIR.Efficiency', 1)]}
