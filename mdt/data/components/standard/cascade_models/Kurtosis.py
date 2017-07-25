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
    inits = {'Kurtosis': [('Kurtosis.theta', 'Tensor.theta'),
                          ('Kurtosis.phi', 'Tensor.phi'),
                          ('Kurtosis.psi', 'Tensor.psi'),
                          ('Kurtosis.d', 'Tensor.d'),
                          ('Kurtosis.dperp0', 'Tensor.dperp0'),
                          ('Kurtosis.dperp1', 'Tensor.dperp1')]}

