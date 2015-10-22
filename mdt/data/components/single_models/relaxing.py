from mdt.components_loader import CompartmentModelsLoader
from mdt.dmri_composite_model import DMRISingleModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lc = CompartmentModelsLoader().load


class TM(DMRISingleModelBuilder):

    name = 'TM'
    description = 'Models TM.'
    model_listing = (lc('ExpT1DecTM'),)
