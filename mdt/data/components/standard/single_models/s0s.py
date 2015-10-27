from mdt.components_loader import CompartmentModelsLoader
from mdt.models.single import DMRISingleModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lc = CompartmentModelsLoader().load


class S0(DMRISingleModelBuilder):

    name = 's0'
    description = 'Models the unweighted signal (aka. b0).'
    model_listing = (lc('S0'),)