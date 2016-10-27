from mdt.models.single import DMRISingleModelConfig
from mot.model_building.parameter_functions.dependencies import SimpleAssignment, AbstractParameterDependency

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

    @property
    def fixed(self):
        return True

    @property
    def has_side_effects(self):
        return False


class NODDI(DMRISingleModelConfig):

    ex_vivo_suitable = False
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
             'Ball.d': 3.0e-9}

    dependencies = (
        ('NODDI_EC.dperp0', NODDITortuosityParameterDependency('NODDI_EC.d', 'w_ec.w', 'w_ic.w')),
        ('NODDI_EC.kappa', SimpleAssignment('NODDI_IC.kappa')),
        ('NODDI_EC.theta', SimpleAssignment('NODDI_IC.theta')),
        ('NODDI_EC.phi', SimpleAssignment('NODDI_IC.phi'))
    )

    post_optimization_modifiers = [
        ('NDI', lambda d: d['w_ic.w'] / (d['w_ic.w'] + d['w_ec.w'])),
        ('ODI', lambda d: d['NODDI_IC.odi'])
    ]


class NODDI2(DMRISingleModelConfig):

    ex_vivo_suitable = False
    description = 'The NODDI model with two IC and EC compartments, non-official adaptation by MDT'

    model_expression = '''
            S0 * (
                  (Weight(w_ic0) * NODDI_IC(NODDI_IC0)) +
                  (Weight(w_ec0) * NODDI_EC(NODDI_EC0)) +

                  (Weight(w_ic1) * NODDI_IC(NODDI_IC1)) +
                  (Weight(w_ec1) * NODDI_EC(NODDI_EC1)) +

                  (Weight(w_csf) * Ball)
            )
        '''

    fixes = {'NODDI_IC0.d': 1.7e-9,
             'NODDI_IC0.R': 0.0,
             'NODDI_IC1.d': 1.7e-9,
             'NODDI_IC1.R': 0.0,
             'NODDI_EC0.d': 1.7e-9,
             'NODDI_EC1.d': 1.7e-9,
             'Ball.d': 3.0e-9}

    dependencies = (
        ('NODDI_EC0.dperp0', NODDITortuosityParameterDependency('NODDI_EC0.d', 'w_ec0.w', 'w_ic0.w')),
        ('NODDI_IC0.kappa', SimpleAssignment('((w_ic0.w + w_ec0.w) >= {cutoff}) * NODDI_IC0.kappa'.format(cutoff=0.01),
                                             fixed=False)),
        ('NODDI_EC0.kappa', SimpleAssignment('NODDI_IC0.kappa')),
        ('NODDI_EC0.theta', SimpleAssignment('NODDI_IC0.theta')),
        ('NODDI_EC0.phi', SimpleAssignment('NODDI_IC0.phi')),

        ('NODDI_EC1.dperp0', NODDITortuosityParameterDependency('NODDI_EC1.d', 'w_ec1.w', 'w_ic1.w')),
        ('NODDI_IC1.kappa', SimpleAssignment('((w_ic1.w + w_ec1.w) >= {cutoff}) * NODDI_IC1.kappa'.format(cutoff=0.01),
                                             fixed=False)),
        ('NODDI_EC1.kappa', SimpleAssignment('NODDI_IC1.kappa')),
        ('NODDI_EC1.theta', SimpleAssignment('NODDI_IC1.theta')),
        ('NODDI_EC1.phi', SimpleAssignment('NODDI_IC1.phi')),
    )

    post_optimization_modifiers = (
        ('NDI0', lambda d: d['w_ic0.w'] / (d['w_ic0.w'] + d['w_ec0.w'])),
        ('ODI0', lambda d: d['NODDI_IC0.odi']),
        ('NDI1', lambda d: d['w_ic1.w'] / (d['w_ic1.w'] + d['w_ec1.w'])),
        ('ODI1', lambda d: d['NODDI_IC1.odi'])
    )
