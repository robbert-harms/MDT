from mdt.components_loader import CompartmentModelsLoader
from mdt.models.single import DMRISingleModelBuilder
from mot.model_building.parameter_functions.dependencies import SimpleAssignment

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lc = CompartmentModelsLoader().load

class ActiveAx(DMRISingleModelBuilder):

    name = 'ActiveAx'
    ex_vivo_suitable = False
    description = 'The standard ActiveAx model'

    model_listing = (lc('S0'),
                     ((lc('Weight', 'Wic'),
                       lc('CylinderGPD').fix('d', 1.7e-9),
                       '*'),
                      (lc('Weight', 'Wec'),
                       lc('Zeppelin').fix('d', 1.7e-9),
                       '*'),
                      (lc('Weight', 'Wcsf'),
                       lc('Ball').fix('d', 3e-9),
                       '*'),
                      '+'),
                     '*')

    dependencies = (('Zeppelin.dperp0', SimpleAssignment('Zeppelin.d * (wec.w / (wec.w + wic.w))')),)
    post_optimization_modifiers = ()