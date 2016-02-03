from mdt.models.single import DMRISingleModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Tensor(DMRISingleModelConfig):

    name = 'Tensor'
    ex_vivo_suitable = False
    description = 'The standard Tensor model with in vivo defaults.'
    model_expression = '''
        S0 * Tensor
    '''
    inits = {'Tensor.d': 1.7e-9,
             'Tensor.dperp0': 1.7e-10,
             'Tensor.dperp1': 1.7e-10}


class TensorExVivo(Tensor):

    name = 'Tensor-ExVivo'
    in_vivo_suitable = False
    description = 'The standard Tensor model with ex vivo defaults.'
    inits = {'Tensor.d': 0.6e-9,
             'Tensor.dperp0': 0.6e-10,
             'Tensor.dperp1': 0.6e-10}


class TensorT2(Tensor):

    name = 'Tensor-T2'
    description = 'The Tensor model with in vivo defaults and extra T2 scaling.'
    model_expression = '''
            S0 * ExpT2Dec * Tensor
        '''


class TensorT2ExVivo(TensorExVivo):

    name = 'Tensor-ExVivo-T2'
    in_vivo_suitable = False
    description = 'The Tensor model with ex vivo defaults and extra T2 scaling.'
    model_expression = '''
            S0 * ExpT2Dec * Tensor
        '''
