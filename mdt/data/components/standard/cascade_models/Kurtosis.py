from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-07-25'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class Kurtosis(CascadeTemplate):

    models = ('Tensor (Cascade)',
              'Kurtosis')

    inits = {'Kurtosis': {
        'KurtosisTensor.theta': 'Tensor.theta',
        'KurtosisTensor.phi': 'Tensor.phi',
        'KurtosisTensor.psi': 'Tensor.psi',
        'KurtosisTensor.d': 'Tensor.d',
        'KurtosisTensor.dperp0': 'Tensor.dperp0',
        'KurtosisTensor.dperp1': 'Tensor.dperp1'}
    }
