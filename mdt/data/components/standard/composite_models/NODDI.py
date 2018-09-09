from mdt import CompositeModelTemplate
from mdt.lib.post_processing import NODDIMeasures

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI(CompositeModelTemplate):
    """The NODDI Watson model"""

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

    extra_optimization_maps = [NODDIMeasures.noddi_watson_extra_optimization_maps]
    extra_sampling_maps = [NODDIMeasures.noddi_watson_extra_sampling_maps]


class NODDIDA(NODDI):
    """The NODDIDA model, NODDI without the Ball compartment and without fixing parameters."""

    model_expression = '''
        S0 * ((Weight(w_ic) * NODDI_IC) +
              (Weight(w_ec) * NODDI_EC)
              )
    '''
    fixes = {'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}


class BinghamNODDI(CompositeModelTemplate):
    """The Bingham NODDI model"""

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_ic) * BinghamNODDI_IN) +
              (Weight(w_ec) * BinghamNODDI_EN)
              )
    '''

    fixes = {'BinghamNODDI_IN.d': 1.7e-9,
             'BinghamNODDI_EN.d': 1.7e-9,
             'Ball.d': 3.0e-9,
             'BinghamNODDI_EN.dperp0': 'BinghamNODDI_EN.d * (isnan(w_ec.w / (w_ec.w + w_ic.w)) ? '
                                       '                     0 : (w_ec.w / (w_ec.w + w_ic.w)))',
             'BinghamNODDI_EN.kappa': 'BinghamNODDI_IN.kappa',
             'BinghamNODDI_EN.beta':  'BinghamNODDI_IN.beta',
             'BinghamNODDI_EN.theta': 'BinghamNODDI_IN.theta',
             'BinghamNODDI_EN.phi':   'BinghamNODDI_IN.phi',
             'BinghamNODDI_EN.psi':   'BinghamNODDI_IN.psi'
             }

    extra_prior = 'return BinghamNODDI_IN.kappa >= BinghamNODDI_IN.beta;'

    extra_optimization_maps = [NODDIMeasures.noddi_bingham_extra_optimization_maps]
    extra_sampling_maps = [NODDIMeasures.noddi_bingham_extra_sampling_maps]
