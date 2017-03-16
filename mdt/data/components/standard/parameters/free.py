"""Definitions of the free parameters.

The free parameters are meant to be used for parameters that one wants to optimize. They can be fixed to a certain
value to disable them from being optimized in a given situation, but they remain classified as 'optimizable' parameters.

Please choose the parameter type for a model and parameter carefully since the type signifies how the parameter and
its data are handled during model construction.
"""

import numpy as np
from mdt.components_config.parameters import FreeParameterConfig
from mot.model_building.parameter_functions.priors import UniformWithinBoundsPrior
from mot.model_building.parameter_functions.proposals import GaussianProposal, CircularGaussianProposal
from mot.model_building.parameter_functions.sample_statistics import CircularGaussianPSS
from mot.model_building.parameter_functions.transformations import ClampTransform, AbsModPiTransform, \
    SinSqrClampTransform, CosSqrClampTransform

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class s0(FreeParameterConfig):

    init_value = 1e4
    lower_bound = 1e-5
    upper_bound = 1e10
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(std=10.0)


class T1(FreeParameterConfig):

    init_value = 0.02
    lower_bound = 1e-5
    upper_bound = 4.0
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(0.0001)


class T2(FreeParameterConfig):

    init_value = 0.01
    lower_bound = 1e-5
    upper_bound = 2.0
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(0.0001)


class T2_star(FreeParameterConfig):

    init_value = 0.01
    lower_bound = 0.0
    upper_bound = 1.0
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(0.0001)


class E1(FreeParameterConfig):
    """This parameter is defined *only* for linear decay T1 fitting in GRE data *with* TR constant.

    This parameter is also defined in the SSFP equation. However, in SSFP this parameter is from the protocol (!)
        E1 = exp( -TR / T1 ).
    After estimation of this parameter, T1 can be recovered by applying the next equation:
        -TR / log( E1 ).
    """

    init_value = 0.37
    lower_bound = 0.0
    upper_bound = 1.0
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(0.0001)


class R1(FreeParameterConfig):
    """R1 = 1/T1, for linear T1Dec or other models. """

    init_value = 2
    lower_bound = 0.25
    upper_bound = 100.0
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(0.0001)


class R2(FreeParameterConfig):
    """R2 = 1/T2, for linear T2Dec or other models."""

    init_value = 5
    lower_bound = 0.5
    upper_bound = 500.0
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(0.0001)


class R2s(FreeParameterConfig):
    """R2s = 1/T2s, for lineaR T2sDec or other models."""

    init_value = 10
    lower_bound = 1
    upper_bound = 50.0
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(0.0001)


class theta(FreeParameterConfig):

    init_value = 1 / 2.0 * np.pi
    lower_bound = 0
    upper_bound = np.pi
    parameter_transform = AbsModPiTransform()
    sampling_proposal = CircularGaussianProposal(np.pi, 0.1)
    sampling_prior = UniformWithinBoundsPrior()
    sampling_statistics = CircularGaussianPSS()


class phi(FreeParameterConfig):

    init_value = 1 / 2.0 * np.pi
    lower_bound = 0
    upper_bound = np.pi
    parameter_transform = AbsModPiTransform()
    sampling_proposal = CircularGaussianProposal(np.pi, 0.1)
    sampling_prior = UniformWithinBoundsPrior()
    sampling_statistics = CircularGaussianPSS()


class psi(FreeParameterConfig):

    init_value = 1 / 2.0 * np.pi
    lower_bound = 0
    upper_bound = np.pi
    parameter_transform = AbsModPiTransform()
    sampling_proposal = CircularGaussianProposal(np.pi, 0.5)
    sampling_prior = UniformWithinBoundsPrior()
    sampling_statistics = CircularGaussianPSS()


class d(FreeParameterConfig):

    init_value = 1.7e-9
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1e-10)


class dperp0(FreeParameterConfig):

    init_value = 1.7e-10
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(5e-10)


class dperp1(FreeParameterConfig):

    init_value = 1.7e-11
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(5e-10)


class R(FreeParameterConfig):

    init_value = 2.0e-6
    lower_bound = 1e-6
    upper_bound = 20e-6
    parameter_transform = CosSqrClampTransform()
    sampling_proposal = GaussianProposal(1e-7)


class kappa(FreeParameterConfig):

    init_value = 1
    lower_bound = 1e-5
    upper_bound = 2 * np.pi
    parameter_transform = CosSqrClampTransform()
    sampling_proposal = GaussianProposal(0.01)


# for use in the GDRCylinder model
class gamma_k(FreeParameterConfig):

    init_value = 1
    lower_bound = 0
    upper_bound = 20
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1.0)


# for use in the GDRCylinder model
class gamma_beta(FreeParameterConfig):

    init_value = 1
    lower_bound = 1.0e-7
    upper_bound = 3.0e-7
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1e-7)


# for use in the GDRCylinder model
class gamma_nmr_cyl(FreeParameterConfig):

    init_value = 5
    lower_bound = 1
    upper_bound = 10
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1)

# for use in ExpT1DecSTEAM model. It is assumed for ex-vivo values. For in-vivo use d instead.
class d_exvivo(FreeParameterConfig):

    init_value = 5.0e-10
    lower_bound = 0.0
    upper_bound = 1.0e-8
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1e-11)
