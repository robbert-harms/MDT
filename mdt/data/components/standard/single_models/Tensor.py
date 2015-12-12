from mdt.components_loader import CompartmentModelsLoader
from mdt.models.single import DMRISingleModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lc = CompartmentModelsLoader().load


class Tensor(DMRISingleModelBuilder):

    name = 'Tensor'
    ex_vivo_suitable = False
    description = 'The standard Tensor model with in vivo defaults.'
    model_listing = (lc('S0'),
                     lc('Tensor').init('d', 1.7e-9)
                                 .init('dperp0', 1.7e-10)
                                 .init('dperp1', 1.7e-10),
                     '*')


class TensorExVivo(DMRISingleModelBuilder):

    name = 'Tensor-ExVivo'
    in_vivo_suitable = False
    description = 'The standard Tensor model with ex vivo defaults.'
    model_listing = (lc('S0'),
                     lc('Tensor').init('d', 0.6e-9)
                                 .init('dperp0', 0.6e-10)
                                 .init('dperp1', 0.6e-10),
                     '*')


class TensorT2(DMRISingleModelBuilder):

    name = 'Tensor-T2'
    ex_vivo_suitable = False
    description = 'The Tensor model with in vivo defaults and extra T2 scaling.'
    model_listing = (lc('S0'),
                     lc('ExpT2Dec'),
                     lc('Tensor').init('d', 1.7e-9)
                                 .init('dperp0', 1.7e-10)
                                 .init('dperp1', 1.7e-10),
                     '*')


class TensorT2ExVivo(DMRISingleModelBuilder):

    name = 'Tensor-ExVivo-T2'
    in_vivo_suitable = False
    description = 'The Tensor model with ex vivo defaults and extra T2 scaling.'
    model_listing = (lc('S0'),
                     lc('ExpT2Dec'),
                     lc('Tensor').init('d', 0.6e-9)
                                 .init('dperp0', 0.6e-10)
                                 .init('dperp1', 0.6e-10),
                     '*')
