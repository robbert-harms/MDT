import numpy as np

from mdt.components_loader import bind_function
from mdt.models.single import DMRISingleModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Tensor(DMRISingleModelConfig):

    ex_vivo_suitable = False
    description = 'The standard Tensor model with in vivo defaults.'
    model_expression = '''
        S0 * Tensor
    '''
    inits = {'Tensor.d': 1.7e-9,
             'Tensor.dperp0': 1.7e-10,
             'Tensor.dperp1': 1.7e-10}

    @bind_function
    def _get_suitable_volume_indices(self, problem_data):
        protocol = problem_data.protocol
        unweighted_threshold = 25e6  # in SI units of s/m^2

        if protocol.has_column('g') and protocol.has_column('b'):
            protocol_indices = protocol.get_indices_bval_in_range(start=0, end=1.5e9 + 0.1e9)
            protocol_indices = np.append(protocol_indices, protocol.get_unweighted_indices(unweighted_threshold))
        else:
            protocol_indices = list(range(protocol.length))

        return np.unique(protocol_indices)


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
