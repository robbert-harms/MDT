import numpy as np

from mdt.components_loader import bind_function
from mdt.models.single import DMRISingleModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class S0(DMRISingleModelConfig):

    description = 'Models the unweighted signal (aka. b0).'
    model_expression = 'S0'

    @bind_function
    def _get_suitable_volume_indices(self, problem_data):
        protocol = problem_data.protocol
        protocol_indices = np.array([], dtype=np.int64)
        unweighted_threshold = 25e6 # in SI units of s/m^2

        if protocol.has_column('g') and protocol.has_column('b'):
            protocol_indices = np.append(protocol_indices, protocol.get_unweighted_indices(unweighted_threshold))
        else:
            protocol_indices = list(range(protocol.length))

        return np.unique(protocol_indices)
