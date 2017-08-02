from mdt.component_templates.cascade_models import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-07-25'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class Kurtosis(CascadeTemplate):

    description = 'Initializes the primary directions using a Tensor estimate.'
    models = ('Tensor (Cascade)',
              'Kurtosis')
