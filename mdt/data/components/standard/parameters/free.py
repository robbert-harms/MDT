import numpy as np
from mdt.models.parameters import FreeParameterConfig
from mot.model_building.parameter_functions.priors import AbsSinHalfPrior, AbsSinPrior
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mot.model_building.parameter_functions.sample_statistics import CircularGaussianPSS
from mot.model_building.parameter_functions.transformations import ClampTransform, AbsModPiTransform, \
    SinSqrClampTransform, CosSqrClampTransform

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


"""Defines the free parameters.

The free parameters are meant to be used for parameters that one wants to optimize. They can be fixed to a certain
value to disable them from being optimized in a given situation, but they remain 'optimable' parameters.

Please choose the parameter type for a model and parameter carefully since the type signifies how the parameter and
its data are handled during model construction.
"""


class s0(FreeParameterConfig):

    name = 's0'
    init_value = 1e4
    lower_bound = 1
    upper_bound = 1e10
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(std=25.0)
    perturbation_function = lambda v: np.clip(v + np.random.normal(scale=1e3, size=v.shape), 1, 1e10)


class T1(FreeParameterConfig):

    name = 'T1'
    init_value = 0.03
    lower_bound = 0.0
    upper_bound = 4.0
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(0.0001)


class T2(FreeParameterConfig):

    name = 'T2'
    init_value = 0.01
    lower_bound = 0.0
    upper_bound = 2.0
    parameter_transform = ClampTransform()
    sampling_proposal = GaussianProposal(0.0001)


class theta(FreeParameterConfig):

    name = 'theta'
    init_value = 1 / 2.0 * np.pi
    lower_bound = 0
    upper_bound = np.pi
    parameter_transform = AbsModPiTransform()
    sampling_proposal = GaussianProposal(0.02)
    sampling_prior = AbsSinHalfPrior()
    sampling_statistics = CircularGaussianPSS()
    perturbation_function = lambda v: v + np.random.normal(scale=0.1, size=v.shape)


class phi(FreeParameterConfig):

    name = 'phi'
    init_value = 1 / 2.0 * np.pi
    lower_bound = 0
    upper_bound = np.pi
    parameter_transform = AbsModPiTransform()
    sampling_proposal = GaussianProposal(0.02)
    sampling_prior = AbsSinPrior()
    sampling_statistics = CircularGaussianPSS()
    perturbation_function = lambda v: v + np.random.normal(scale=0.1, size=v.shape)


class psi(FreeParameterConfig):

    name = 'psi'
    init_value = 1 / 2.0 * np.pi
    lower_bound = 0
    upper_bound = np.pi
    parameter_transform = AbsModPiTransform()
    sampling_proposal = GaussianProposal(0.02)
    sampling_prior = AbsSinPrior()
    sampling_statistics = CircularGaussianPSS()
    perturbation_function = lambda v: v + np.random.normal(scale=0.1, size=v.shape)


class d(FreeParameterConfig):

    name = 'd'
    init_value = 1.7e-9
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1e-14)


class dperp0(FreeParameterConfig):

    name = 'dperp0'
    init_value = 1.7e-10
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1e-15)


class dperp1(FreeParameterConfig):

    name = 'dperp1'
    init_value = 1.7e-11
    lower_bound = 0
    upper_bound = 1.0e-8
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1e-15)


class R(FreeParameterConfig):

    name = 'R'
    init_value = 2.0e-6
    lower_bound = 1e-6
    upper_bound = 20e-6
    parameter_transform = CosSqrClampTransform()
    sampling_proposal = GaussianProposal(1e-6)


class kappa(FreeParameterConfig):

    name = 'kappa'
    init_value = 1
    lower_bound = 1e-5
    upper_bound = 2 * np.pi
    parameter_transform = CosSqrClampTransform()
    sampling_proposal = GaussianProposal(0.1)


# for use in the GDRCylinder model
class gamma_k(FreeParameterConfig):

    name = 'gamma_k'
    init_value = 1
    lower_bound = 0
    upper_bound = 20
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1.0)


# for use in the GDRCylinder model
class gamma_beta(FreeParameterConfig):

    name = 'gamma_beta'
    init_value = 1
    lower_bound = 1.0e-7
    upper_bound = 3.0e-7
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1e-7)


# for use in the GDRCylinder model
class gamma_nmr_cyl(FreeParameterConfig):

    name = 'gamma_nmr_cyl'
    init_value = 5
    lower_bound = 1
    upper_bound = 10
    parameter_transform = SinSqrClampTransform()
    sampling_proposal = GaussianProposal(1)
