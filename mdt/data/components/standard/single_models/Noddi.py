from mdt.models.single import DMRISingleModelConfig
from mot.model_building.parameter_functions.dependencies import SimpleAssignment, AbstractParameterDependency

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


"""
The Noddi compartments and models with CamelCase are being deprecated. Please use the all-capital version (NODDI)
instead in new works.
"""


class NoddiTortuosityParameterDependency(AbstractParameterDependency):

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

    @property
    def fixed(self):
        return True

    @property
    def has_side_effects(self):
        return False


class Noddi(DMRISingleModelConfig):

    ex_vivo_suitable = False
    description = 'The standard Noddi (NODDI) model'

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_ic) * Noddi_IC) +
              (Weight(w_ec) * Noddi_EC)
              )
    '''

    fixes = {'Noddi_IC.d': 1.7e-9,
             'Noddi_IC.R': 0.0,
             'Noddi_EC.d': 1.7e-9,
             'Ball.d': 3.0e-9}

    dependencies = (
        ('Noddi_EC.dperp0', NoddiTortuosityParameterDependency('Noddi_EC.d', 'w_ec.w', 'w_ic.w')),
        ('Noddi_EC.kappa', SimpleAssignment('Noddi_IC.kappa')),
        ('Noddi_EC.theta', SimpleAssignment('Noddi_IC.theta')),
        ('Noddi_EC.phi', SimpleAssignment('Noddi_IC.phi'))
    )

    post_optimization_modifiers = [
        ('NDI', lambda d: d['w_ic.w'] / (d['w_ic.w'] + d['w_ec.w'])),
        ('ODI', lambda d: d['Noddi_IC.odi'])
    ]

