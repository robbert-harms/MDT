from mdt.components_config.composite_models import DMRICompositeModelConfig
from mot.model_building.parameter_functions.dependencies import AbstractParameterDependency

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDITortuosityParameterDependency(AbstractParameterDependency):

    def __init__(self, d, w_ec, w_ic, ):
        self._d = d
        self._w_ec = w_ec
        self._w_ic = w_ic

    @property
    def pre_transform_code(self):
        return '''
            mot_float_type _tortuosity_mult_{d} = {w_ec} / ({w_ec} + {w_ic});
            if(!isnormal(_tortuosity_mult_{d})){{
                _tortuosity_mult_{d} = 0.01;
            }}
        '''.format(d=self._d, w_ec=self._w_ec, w_ic=self._w_ic)

    @property
    def assignment_code(self):
        return '{d} * _tortuosity_mult_{d}'.format(d=self._d)


class NODDI(DMRICompositeModelConfig):

    description = 'The standard NODDI model'

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_ic) * NODDI_IC) +
              (Weight(w_ec) * NODDI_EC)
              )
    '''

    fixes = {'NODDI_IC.d': 1.7e-9,
             'NODDI_IC.R': 0.0,
             'NODDI_EC.d': 1.7e-9,
             'Ball.d': 3.0e-9,
             'NODDI_EC.dperp0': NODDITortuosityParameterDependency('NODDI_EC.d', 'w_ec.w', 'w_ic.w'),
             'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}

    post_optimization_modifiers = [
        ('NDI', lambda d: d['w_ic.w'] / (d['w_ic.w'] + d['w_ec.w'])),
        ('ODI', lambda d: d['NODDI_IC.odi'])
    ]

