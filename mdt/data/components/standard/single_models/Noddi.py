from mdt.models.single import DMRISingleModelConfig
from mot.model_building.parameter_functions.dependencies import SimpleAssignment

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Noddi(DMRISingleModelConfig):

    name = 'Noddi'
    ex_vivo_suitable = False
    description = 'The standard Noddi (NODDI) model'

    model_expression = '''
        S0 * ((Weight(w_ic) * Noddi_IC) +
              (Weight(w_ec) * Noddi_EC) +
              (Weight(w_csf) * Ball))
    '''

    fixes = {'Noddi_IC.d': 1.7e-9,
             'Noddi_IC.R': 0.0,
             'Noddi_EC.d': 1.7e-9,
             'Ball.d': 3.0e-9}

    dependencies = (
        ('Noddi_EC.dperp0', SimpleAssignment('Noddi_EC.d * (w_ec.w / (1 - w_csf.w + {eps}))'.format(eps=1e-5),
                                             fixed=False)), # actually Fixed should be true, but for some reason
                                                              # this does not work in the combination of double precision
                                                              # and the AMD R9 280x card. It does however work when
                                                              # we set -cl-opt-disable. Therefore, my idea is that it
                                                              # has something to do with the AMD kernel optimizer.
        ('Noddi_IC.kappa', SimpleAssignment('((1 - w_csf.w) >= {cutoff}) * Noddi_IC.kappa'.format(cutoff=0.01),
                                            fixed=False)),
        ('Noddi_EC.kappa', SimpleAssignment('Noddi_IC.kappa')),
        ('Noddi_EC.theta', SimpleAssignment('Noddi_IC.theta')),
        ('Noddi_EC.phi', SimpleAssignment('Noddi_IC.phi'))
    )

    post_optimization_modifiers = [
        ('NDI', lambda d: d['w_ic.w'] / (d['w_ic.w'] + d['w_ec.w'])),
        ('ODI', lambda d: d['Noddi_IC.odi'])
    ]


class Noddi2(DMRISingleModelConfig):

    name = 'Noddi2'
    ex_vivo_suitable = False
    description = 'The Noddi model with two IC and EC compartments'

    model_expression = '''
            S0 * (
                  (Weight(w_ic0) * Noddi_IC(Noddi_IC0)) +
                  (Weight(w_ec0) * Noddi_EC(Noddi_EC0)) +

                  (Weight(w_ic1) * Noddi_IC(Noddi_IC1)) +
                  (Weight(w_ec1) * Noddi_EC(Noddi_EC1)) +

                  (Weight(w_csf) * Ball)
            )
        '''

    fixes = {'Noddi_IC0.d': 1.7e-9,
             'Noddi_IC0.R': 0.0,
             'Noddi_IC1.d': 1.7e-9,
             'Noddi_IC1.R': 0.0,
             'Noddi_EC0.d': 1.7e-9,
             'Noddi_EC1.d': 1.7e-9,
             'Ball.d': 3.0e-9}

    dependencies = (
        ('Noddi_EC0.dperp0', SimpleAssignment('Noddi_EC0.d * (w_ec0.w / (w_ic0.w + w_ec0.w + {eps}))'.format(eps = 1e-5))),
        ('Noddi_IC0.kappa', SimpleAssignment('((w_ic0.w + w_ec0.w) >= {cutoff}) * Noddi_IC0.kappa'.format(cutoff = 0.01),
                                             fixed=False)),
        ('Noddi_EC0.kappa', SimpleAssignment('Noddi_IC0.kappa')),
        ('Noddi_EC0.theta', SimpleAssignment('Noddi_IC0.theta')),
        ('Noddi_EC0.phi', SimpleAssignment('Noddi_IC0.phi')),

        ('Noddi_EC1.dperp0', SimpleAssignment('Noddi_EC1.d * (w_ec1.w / (w_ic1.w + w_ec1.w + {eps}))'.format(eps = 1e-5))),
        ('Noddi_IC1.kappa', SimpleAssignment('((w_ic1.w + w_ec1.w) >= {cutoff}) * Noddi_IC1.kappa'.format(cutoff = 0.01),
                                             fixed=False)),
        ('Noddi_EC1.kappa', SimpleAssignment('Noddi_IC1.kappa')),
        ('Noddi_EC1.theta', SimpleAssignment('Noddi_IC1.theta')),
        ('Noddi_EC1.phi', SimpleAssignment('Noddi_IC1.phi')),
    )

    post_optimization_modifiers = (
        ('NDI0', lambda d: d['w_ic0.w'] / (d['w_ic0.w'] + d['w_ec0.w'])),
        ('ODI0', lambda d: d['Noddi_IC0.odi']),
        ('NDI1', lambda d: d['w_ic1.w'] / (d['w_ic1.w'] + d['w_ec1.w'])),
        ('ODI1', lambda d: d['Noddi_IC1.odi'])
    )
