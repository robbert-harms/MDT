from mdt import CompositeModelTemplate
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI(CompositeModelTemplate):

    description = 'The standard NODDI model'

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_ic) * NODDI_IC) +
              (Weight(w_ec) * NODDI_EC)
              )
    '''

    fixes = {'NODDI_IC.d': 1.7e-9,
             'NODDI_EC.d': 1.7e-9,
             'Ball.d': 3.0e-9,
             'NODDI_EC.dperp0': 'NODDI_EC.d * (isnan(w_ec.w / (w_ec.w + w_ic.w)) ? 0 : (w_ec.w / (w_ec.w + w_ic.w)))',
             'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}

    extra_optimization_maps = [
        lambda d: {'NDI': d['w_ic.w'] / (d['w_ic.w'] + d['w_ec.w']),
                   'ODI': np.arctan2(1.0, d['NODDI_IC.kappa']) * 2 / np.pi}
    ]
    extra_sampling_maps = [
        lambda samples: {'w_ic.w': np.mean(samples['w_ic.w'], axis=1),
                         'w_ic.w.std': np.std(samples['w_ic.w'], axis=1),
                         'NDI': np.mean(samples['w_ic.w'] / (samples['w_ic.w'] + samples['w_ec.w']), axis=1),
                         'NDI.std': np.std(samples['w_ic.w'] / (samples['w_ic.w'] + samples['w_ec.w']), axis=1),
                         'ODI': np.mean(np.arctan2(1.0, samples['NODDI_IC.kappa']) * 2 / np.pi, axis=1),
                         'ODI.std': np.std(np.arctan2(1.0, samples['NODDI_IC.kappa']) * 2 / np.pi, axis=1)
                         }
    ]


class NODDIDA(NODDI):

    description = 'The NODDIDA model, NODDI without the Ball compartment and without fixing parameters.'

    model_expression = '''
        S0 * ((Weight(w_ic) * NODDI_IC) +
              (Weight(w_ec) * NODDI_EC)
              )
    '''

    fixes = {'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}
