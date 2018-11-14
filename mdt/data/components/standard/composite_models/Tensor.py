from mdt import CompositeModelTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Tensor(CompositeModelTemplate):
    model_expression = '''
        S0 * Tensor
    '''
    inits = {'Tensor.d': 1.7e-9,
             'Tensor.dperp0': 1.7e-10,
             'Tensor.dperp1': 1.7e-10}
    volume_selection = {'b': [(0, 1.5e9 + 0.1e9)]}


class NonParametricTensor(CompositeModelTemplate):
    model_expression = '''
        S0 * SymmetricNonParametricTensor(Tensor)
    '''
    volume_selection = {'b': [(0, 1.5e9 + 0.1e9)]}
