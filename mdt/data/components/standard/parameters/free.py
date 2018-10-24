"""Definitions of the free parameters.

The free parameters are meant to be used for parameters that one wants to optimize. They can be fixed to a certain
value to disable them from being optimized in a given situation, but they remain classified as 'optimizable' parameters.
"""

import numpy as np
from mdt import FreeParameterTemplate
from mdt.model_building.parameter_functions.priors import UniformWithinBoundsPrior, ARDBeta, ARDGaussian

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class s0(FreeParameterTemplate):

    init_value = 1e4
    lower_bound = 0
    upper_bound = 1e10
    parameter_transform = 'Clamp'
    sampling_proposal_std = 10.0


class w(FreeParameterTemplate):

    init_value = 0.5
    lower_bound = 0
    upper_bound = 1
    parameter_transform = 'CosSqrClamp'
    sampling_proposal_std = 0.01
    sampling_prior = UniformWithinBoundsPrior()
    numdiff_info = {'scale_factor': 10}


class w_ard_beta(w):
    """Subclasses the weight to add a Beta prior for in use with Automatic Relevance Detection during sample."""
    sampling_prior = ARDBeta()


class w_ard_gaussian(w):
    """Subclasses the weight to add a Gaussian prior for in use with Automatic Relevance Detection during sample."""
    sampling_prior = ARDGaussian()


class T1(FreeParameterTemplate):

    init_value = 0.02
    lower_bound = 1e-5
    upper_bound = 4.0
    parameter_transform = 'Clamp'


class T2(FreeParameterTemplate):

    init_value = 0.01
    lower_bound = 1e-5
    upper_bound = 2.0
    parameter_transform = 'Clamp'


class T2_star(FreeParameterTemplate):

    init_value = 0.01
    lower_bound = 0.0
    upper_bound = 1.0
    parameter_transform = 'Clamp'


class E1(FreeParameterTemplate):
    """This parameter is defined *only* for linear decay T1 fitting in GRE data *with* TR constant.

    This parameter is also defined in the SSFP equation. However, in SSFP this parameter is from the protocol (!)
        E1 = exp( -TR / T1 ).
    After estimation of this parameter, T1 can be recovered by applying the next equation:
        -TR / log( E1 ).
    """
    init_value = 0.37
    lower_bound = 0.0
    upper_bound = 1.0
    parameter_transform = 'Clamp'


class R1(FreeParameterTemplate):
    """R1 = 1/T1, for linear T1Dec or other models. """

    init_value = 2
    lower_bound = 0.25
    upper_bound = 100.0
    parameter_transform = 'Clamp'


class R2(FreeParameterTemplate):
    """R2 = 1/T2, for linear T2Dec or other models."""

    init_value = 5
    lower_bound = 0.5
    upper_bound = 500.0
    parameter_transform = 'Clamp'


class R2s(FreeParameterTemplate):
    """R2s = 1/T2s, for lineaR T2sDec or other models."""

    init_value = 10
    lower_bound = 1
    upper_bound = 50.0
    parameter_transform = 'Clamp'


class theta(FreeParameterTemplate):
    """The inclination/polar angle.

    This parameter is limited between [0, pi] but with modulus pi. That is, 0 == pi and this parameter should be
    allowed to wrap around pi.

    """
    init_value = np.pi / 2.0
    lower_bound = 0
    upper_bound = np.pi
    parameter_transform = 'AbsModPi'
    sampling_proposal_std = 0.1
    sampling_proposal_modulus = np.pi
    numdiff_info = {'use_bounds': False}


class phi(FreeParameterTemplate):
    """The azimuth angle.

    We limit this parameter between [0, pi] making us only use (together with theta between [0, pi]) only the right
    hemisphere. This is possible since diffusion is symmetric and works fine during optimization. For calculating the
    numerical derivative we can let phi rotate around 2*pi again.

    During sample the results can clip to pi since the standard formula for transforming spherical coordinates to
    cartesian coordinates defines phi to be in the range [0, 2*pi]. This is both a problem and a blessing. The problem
    is that the samples will not wrap nicely around pi, the blessing is that we prevent a bimodal distribution in phi.
    Not wrapping around pi is not much of a problem though, as the sampler can easily sample only half of a gaussian
    if the optimal parameter is around zero or pi.
    """
    init_value = np.pi / 2.0
    lower_bound = 0
    upper_bound = np.pi
    parameter_transform = 'AbsModPi'
    sampling_proposal_std = 0.1
    sampling_proposal_modulus = np.pi
    numdiff_info = {'use_bounds': False}


class psi(FreeParameterTemplate):
    """The rotation angle for use in cylindrical models.

    This parameter can be used to rotate a vector around another vector, as is for example done in the Tensor model.
    """
    init_value = np.pi / 2.0
    lower_bound = 0
    upper_bound = np.pi
    parameter_transform = 'AbsModPi'
    sampling_proposal_modulus = np.pi
    sampling_proposal_std = 0.1
    numdiff_info = {'use_bounds': False}


class d(FreeParameterTemplate):

    init_value = 1.7e-9
    lower_bound = 1e-11
    upper_bound = 1.0e-8
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class dperp0(FreeParameterTemplate):

    init_value = 1.7e-10
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class dperp1(FreeParameterTemplate):

    init_value = 1.7e-11
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class R(FreeParameterTemplate):
    init_value = 1.0e-6
    lower_bound = 1e-7
    upper_bound = 20e-6
    parameter_transform = 'CosSqrClamp'
    sampling_proposal_std = 1e-7


class kappa(FreeParameterTemplate):
    """The kappa parameter used in the NODDI Watson model.

    The NODDI-Watson model computes the spherical harmonic (SH) coefficients of the Watson distribution with the
    concentration parameter k (kappa) up to the 12th order.

    Truncating at the 12th order gives good approximation for kappa up to 64, as such we define kappa to be between
    zero and 64.
    """
    init_value = 1
    lower_bound = 0
    upper_bound = 64
    parameter_transform = 'CosSqrClamp'
    sampling_proposal_std = 0.01
    numdiff_info = {'max_step': 0.1, 'use_upper_bound': False}


class k1(FreeParameterTemplate):
    """The kappa parameter for the Ball&Racket and NODDI Bingham model"""
    init_value = 1
    lower_bound = 0
    upper_bound = 64
    parameter_transform = 'CosSqrClamp'
    sampling_proposal_std = 0.01
    numdiff_info = {'max_step': 0.1, 'use_upper_bound': False}


class kw(FreeParameterTemplate):
    """We optimize the ratio w = k1/k2 in the Ball&Racket and NODDI Bingham model"""
    init_value = 2
    lower_bound = 1
    upper_bound = 64
    parameter_transform = 'CosSqrClamp'
    sampling_proposal_std = 0.01
    numdiff_info = {'max_step': 0.1, 'use_upper_bound': False}


class d_exvivo(FreeParameterTemplate):
    """For use in ExpT1DecSTEAM model. It assumes ex-vivo values. For in-vivo use ``d`` instead."""
    init_value = 5.0e-10
    lower_bound = 0.0
    upper_bound = 1.0e-8
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-11
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class time_dependent_characteristic_coefficient(FreeParameterTemplate):
    """The time dependent characteristic as used in the TimeDependentZeppelin model. Values are in m^2."""
    init_value = 1e-6
    lower_bound = 1e-7
    upper_bound = 1e-5
    parameter_transform = 'CosSqrClamp'
    sampling_proposal_std = 1e-7


class d_bulk(FreeParameterTemplate):

    init_value = 0.e-9
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


# the following parameters are part of the non-parametric Tensor
# (Tensor in which a upper triangular D matrix is optimized directly)
class Tensor_D_00(FreeParameterTemplate):

    init_value = 0.3e-9
    lower_bound = 0
    upper_bound = 5e-9
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class Tensor_D_11(FreeParameterTemplate):

    init_value = 0.3e-9
    lower_bound = 0
    upper_bound = 5e-9
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class Tensor_D_22(FreeParameterTemplate):

    init_value = 1.2e-9
    lower_bound = 0
    upper_bound = 5e-9
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class Tensor_D_01(FreeParameterTemplate):

    init_value = 0
    lower_bound = -1e-9
    upper_bound = 1e-9
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class Tensor_D_02(FreeParameterTemplate):

    init_value = 0
    lower_bound = -1e-9
    upper_bound = 1e-9
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class Tensor_D_12(FreeParameterTemplate):

    init_value = 0
    lower_bound = -1e-9
    upper_bound = 1e-9
    parameter_transform = 'SinSqrClamp'
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}
