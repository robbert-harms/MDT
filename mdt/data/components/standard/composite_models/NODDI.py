import numpy as np

from mdt import CompositeModelTemplate
from mdt.lib.post_processing import NODDIMeasures

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert@xkls.nl"


class NODDI(CompositeModelTemplate):
    """The NODDI Watson model"""

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_ic) * NODDI_IC) +
              (Weight(w_ec) * NODDI_EC))
    '''

    fixes = {'NODDI_IC.d': 1.7e-9,
             'NODDI_EC.d': 1.7e-9,
             'Ball.d': 3.0e-9,
             'NODDI_EC.dperp0': 'NODDI_EC.d * w_ec.w / (w_ec.w + w_ic.w)',
             'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}

    extra_optimization_maps = [NODDIMeasures.noddi_watson_extra_optimization_maps]
    extra_sampling_maps = [NODDIMeasures.noddi_watson_extra_sampling_maps]


class NODDI_ZhangFormulation(CompositeModelTemplate):
    """The NODDI Watson model using the original formulation from Gary Zhang."""

    model_expression = '''
        S0 * (
                Weight(fiso_neg) * (
                    (Weight(ficvf) * NODDI_IC) +
                    (Weight(ficvf_neg) * NODDI_EC)
                ) +
                (Weight(fiso) * Ball)
        )
    '''

    enforce_weights_sum_to_one = False

    fixes = {'NODDI_IC.d': 1.7e-9,
             'NODDI_EC.d': 1.7e-9,
             'Ball.d': 3.0e-9,
             'fiso_neg.w': '1 - fiso.w',
             'ficvf_neg.w': '1 - ficvf.w',
             'NODDI_EC.dperp0': 'NODDI_EC.d * (1 - ficvf.w)',
             'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}

    extra_optimization_maps = [lambda results: {'ODI': np.arctan2(1.0, results['NODDI_IC.kappa']) * 2 / np.pi}]


class NODDI_Fixed_IC_EC_LT(CompositeModelTemplate):
    """ Fixed IC parallel. EC estimated but less than IC value with constraint. """

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_ic) * NODDI_IC) +
              (Weight(w_ec) * NODDI_EC))
    '''
    fixes = {
             'NODDI_IC.d': 1.7e-9,
             'Ball.d': 3.0e-9,
             'NODDI_EC.dperp0': 'NODDI_EC.d * w_ec.w / (w_ec.w + w_ic.w)',
             'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}
    inits = {
        'NODDI_EC.d': 1.5e-9
    }
    constraints = '''
        constraints[0] = NODDI_EC.d - NODDI_IC.d;
    '''
    extra_optimization_maps = [NODDIMeasures.noddi_watson_extra_optimization_maps]
    extra_sampling_maps = [NODDIMeasures.noddi_watson_extra_sampling_maps]


class NODDI_ExVivo(CompositeModelTemplate):
    """The NODDI Watson model"""

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_stat) * Dot) +
              (Weight(w_ic) * NODDI_IC) +
              (Weight(w_ec) * NODDI_EC))
    '''
    inits = {
        'w_csf.w': 0.01,
        'w_stat.w': 0.01
    }
    fixes = {'NODDI_IC.d': 0.6e-9,
             'Ball.d': 2.0e-9,
             'NODDI_EC.d': 'NODDI_IC.d',
             'NODDI_EC.dperp0': 'NODDI_EC.d * w_ec.w / (w_ec.w + w_ic.w)',
             'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}

    extra_optimization_maps = [NODDIMeasures.noddi_watson_extra_optimization_maps]
    extra_sampling_maps = [NODDIMeasures.noddi_watson_extra_sampling_maps]


class NODDIDA(NODDI):
    """The NODDIDA model, NODDI without the Ball compartment and without fixing parameters.

    This implements the constraint NODDI_EC.d <= NODDI_IC.d as described in [1].

    References:
        1. Nicolas Kunz, et al., Intra- and extra-axonal axial diffusivities in the white matter: Which one is faster?,
            NeuroImage, 2018. DOI: https://doi.org/10.1016/j.neuroimage.2018.07.020.
    """

    model_expression = '''
        S0 * ((Weight(w_ic) * NODDI_IC) +
              (Weight(w_ec) * NODDI_EC))
    '''
    fixes = {'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}

    constraints = '''
        constraints[0] = NODDI_EC.d - NODDI_IC.d;
    '''

    extra_optimization_maps = [NODDIMeasures.noddi_watson_extra_optimization_maps]
    extra_sampling_maps = [NODDIMeasures.noddi_watson_extra_sampling_maps]


class BinghamNODDI_r1(CompositeModelTemplate):
    """The Bingham NODDI model with one intra- and extra-cellular compartments"""

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_in0) * BinghamNODDI_IN(BinghamNODDI_IN0)) +
              (Weight(w_en0) * BinghamNODDI_EN(BinghamNODDI_EN0)))
    '''
    fixes = {
        'BinghamNODDI_IN0.d': 1.7e-9,
        'BinghamNODDI_EN0.d': 1.7e-9,
        'Ball.d': 3.0e-9,
        'BinghamNODDI_EN0.dperp0': 'BinghamNODDI_EN0.d * w_en0.w / (w_en0.w + w_in0.w)',
        'BinghamNODDI_EN0.k1': 'BinghamNODDI_IN0.k1',
        'BinghamNODDI_EN0.kw':  'BinghamNODDI_IN0.kw',
        'BinghamNODDI_EN0.theta': 'BinghamNODDI_IN0.theta',
        'BinghamNODDI_EN0.phi':   'BinghamNODDI_IN0.phi',
        'BinghamNODDI_EN0.psi':   'BinghamNODDI_IN0.psi'
    }
    extra_optimization_maps = [NODDIMeasures.noddi_bingham_extra_optimization_maps]
    extra_sampling_maps = [NODDIMeasures.noddi_bingham_extra_sampling_maps]


class BinghamNODDI_r2(CompositeModelTemplate):
    """The Bingham NODDI model with two intra- and extra-cellular compartments."""

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_in0) * BinghamNODDI_IN(BinghamNODDI_IN0)) +
              (Weight(w_en0) * BinghamNODDI_EN(BinghamNODDI_EN0)) +
              (Weight(w_in1) * BinghamNODDI_IN(BinghamNODDI_IN1)) +
              (Weight(w_en1) * BinghamNODDI_EN(BinghamNODDI_EN1)))
    '''
    fixes = {
        'Ball.d': 3.0e-9,
        'BinghamNODDI_IN0.d': 1.7e-9,
        'BinghamNODDI_EN0.d': 1.7e-9,

        'BinghamNODDI_EN0.dperp0': 'BinghamNODDI_EN0.d * w_en0.w / (w_en0.w + w_in0.w)',
        'BinghamNODDI_EN0.k1': 'BinghamNODDI_IN0.k1',
        'BinghamNODDI_EN0.kw':  'BinghamNODDI_IN0.kw',
        'BinghamNODDI_EN0.theta': 'BinghamNODDI_IN0.theta',
        'BinghamNODDI_EN0.phi':   'BinghamNODDI_IN0.phi',
        'BinghamNODDI_EN0.psi':   'BinghamNODDI_IN0.psi',

        'BinghamNODDI_IN1.d': 1.7e-9,
        'BinghamNODDI_EN1.d': 1.7e-9,
        'BinghamNODDI_EN1.dperp0': 'BinghamNODDI_EN1.d * w_en1.w / (w_en1.w + w_in1.w)',
        'BinghamNODDI_EN1.k1': 'BinghamNODDI_IN1.k1',
        'BinghamNODDI_EN1.kw': 'BinghamNODDI_IN1.kw',
        'BinghamNODDI_EN1.theta': 'BinghamNODDI_IN1.theta',
        'BinghamNODDI_EN1.phi': 'BinghamNODDI_IN1.phi',
        'BinghamNODDI_EN1.psi': 'BinghamNODDI_IN1.psi'
    }
    extra_optimization_maps = [NODDIMeasures.noddi_bingham_extra_optimization_maps]
    extra_sampling_maps = [NODDIMeasures.noddi_bingham_extra_sampling_maps]
