from mdt.models.single import DMRISingleModelConfig
from mot.model_building.parameter_functions.dependencies import SimpleAssignment

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ActiveAx(DMRISingleModelConfig):

    ex_vivo_suitable = False
    description = 'The standard ActiveAx model'
    model_expression = '''
        S0 * ((Weight(w_ic) * CylinderGPD) +
              (Weight(w_ec) * Zeppelin) +
              (Weight(w_csf) * Ball))
    '''
    fixes = {'CylinderGPD.d': 1.7e-9,
             'Zeppelin.d': 1.7e-9,
             'Ball.d': 3.0e-9}
    dependencies = (('Zeppelin.dperp0', SimpleAssignment('Zeppelin.d * (w_ec.w / (w_ec.w + w_ic.w))')),)
