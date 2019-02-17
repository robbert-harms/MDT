"""Definitions of the free parameters.

The free parameters are meant to be used for parameters that one wants to optimize. They can be fixed to a certain
value to disable them from being optimized in a given situation, but they remain classified as 'optimizable' parameters.
"""

from mdt import FreeParameterTemplate
from mdt.component_templates.parameters import PolarAngleParameterTemplate, AzimuthAngleParameterTemplate, \
    RotationalAngleParameterTemplate
from mdt.model_building.parameter_functions.priors import UniformWithinBoundsPrior, ARDBeta, ARDGaussian
from mdt.model_building.parameter_functions.transformations import ScaleTransform

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class s0(FreeParameterTemplate):

    init_value = 1e4
    lower_bound = 0
    upper_bound = 1e10
    sampling_proposal_std = 10.0


class w(FreeParameterTemplate):

    init_value = 0.5
    lower_bound = 0
    upper_bound = 1
    parameter_transform = ScaleTransform(10)
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

    init_value = 0.05
    lower_bound = 1e-5
    upper_bound = 4.0
    parameter_transform = ScaleTransform(1e4)


class T2(FreeParameterTemplate):

    init_value = 0.05
    lower_bound = 1e-5
    upper_bound = 2.0
    parameter_transform = ScaleTransform(1e4)


class R1(FreeParameterTemplate):
    """R1 = 1/T1, for linear T1Dec or other models. """

    init_value = 2
    lower_bound = 0.25
    upper_bound = 100.0
    parameter_transform = ScaleTransform(1e2)


class R2(FreeParameterTemplate):
    """R2 = 1/T2, for linear T2Dec or other models."""

    init_value = 5
    lower_bound = 0.5
    upper_bound = 500.0
    parameter_transform = ScaleTransform(1e2)


class R2s(FreeParameterTemplate):
    """R2s = 1/T2s, for lineaR T2sDec or other models."""

    init_value = 10
    lower_bound = 1
    upper_bound = 50.0
    parameter_transform = ScaleTransform(1e2)


class theta(PolarAngleParameterTemplate):
    """The polar/inclination angle for spherical coordinates.

    We subclass from a special spherical coordinate template class to signal to the composite model we
    want to restrict this parameter between [0, pi], together with phi.
    """


class phi(AzimuthAngleParameterTemplate):
    """The azimuth angle for spherical coordinates.

    We subclass from a special spherical coordinate template class to signal to the composite model we
    want to restrict this parameter between [0, pi], together with theta.
    """


class psi(RotationalAngleParameterTemplate):
    """The rotation angle for use in cylindrical models.

    This parameter can be used to rotate a vector around another vector, as is for example done in the Tensor model.

    This parameter is not part of the spherical coordinate parameters.
    """


class d(FreeParameterTemplate):

    init_value = 1.7e-9
    lower_bound = 1e-12
    upper_bound = 1.0e-8
    parameter_transform = ScaleTransform(1e10)
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class dperp0(FreeParameterTemplate):

    init_value = 1.7e-10
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = ScaleTransform(1e10)
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class dperp1(FreeParameterTemplate):

    init_value = 1.7e-11
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = ScaleTransform(1e10)
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class R(FreeParameterTemplate):
    init_value = 1.0e-6
    lower_bound = 1e-7
    upper_bound = 20e-6
    parameter_transform = ScaleTransform(1e7)
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
    sampling_proposal_std = 0.01
    numdiff_info = {'use_upper_bound': False}


class k1(FreeParameterTemplate):
    """The kappa parameter for the Ball&Racket and NODDI Bingham model"""
    init_value = 1
    lower_bound = 0
    upper_bound = 64
    sampling_proposal_std = 0.01
    numdiff_info = {'use_upper_bound': False}


class kw(FreeParameterTemplate):
    """We optimize the ratio w = k1/k2 in the Ball&Racket and NODDI Bingham model"""
    init_value = 2
    lower_bound = 1
    upper_bound = 64
    sampling_proposal_std = 0.01
    numdiff_info = {'use_upper_bound': False}


class d_exvivo(FreeParameterTemplate):
    """For use in ExpT1DecSTEAM model. It assumes ex-vivo values. For in-vivo use ``d`` instead."""
    init_value = 5.0e-10
    lower_bound = 0.0
    upper_bound = 1.0e-8
    parameter_transform = ScaleTransform(1e10)
    sampling_proposal_std = 1e-11
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}


class d_bulk(FreeParameterTemplate):

    init_value = 0.e-9
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = ScaleTransform(1e10)
    sampling_proposal_std = 1e-10
    numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}
