import numpy as np
from mot.base import CLDataType, ProtocolParameter, FreeParameter, ModelDataParameter
from mot.parameter_functions.priors import AbsSinPrior, AbsSinHalfPrior
from mot.parameter_functions.proposals import GaussianProposal
from mot.parameter_functions.sample_statistics import CircularGaussianPSS
from mot.parameter_functions.transformations import ClampTransform, \
    AbsModPiTransform, SinSqrClampTransform, CosSqrClampTransform
from mdt.utils import get_bessel_roots


__author__ = 'Robbert Harms'
__date__ = "2014-05-12"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_parameter(param_name):
    """Get a parameter by name.

    These parameters are used in the builtin compartment models and can also be used by the user.

    Please note that these parameters come with default values suitable for most models. Please do not change these
    default values without consideration.

    Args:
        param_name (str): The name of the parameter. For example 'theta'.

    Returns:
        the parameter object for the requested parameter
    """
    param_map = {'g': ProtocolParameter(CLDataType.from_string('model_float4'), 'g'),
                 'G': ProtocolParameter(CLDataType.from_string('model_float'), 'G'),
                 'Delta': ProtocolParameter(CLDataType.from_string('model_float'), 'Delta'),
                 'delta': ProtocolParameter(CLDataType.from_string('model_float'), 'delta'),
                 'TE': ProtocolParameter(CLDataType.from_string('model_float'), 'TE'),
                 'b': ProtocolParameter(CLDataType.from_string('model_float'), 'b'),
                 'q': ProtocolParameter(CLDataType.from_string('model_float'), 'q'),
                 'T1': ProtocolParameter(CLDataType.from_string('model_float'), 'T1'),
                 'GAMMA2_G2_delta2': ProtocolParameter(CLDataType.from_string('model_float'), 'GAMMA2_G2_delta2'),

                 'T2': FreeParameter(CLDataType.from_string('double'), 'T2', False, 0.05, 0.0, 0.8,
                                     parameter_transform=ClampTransform(),  sampling_proposal=GaussianProposal(0.0001)),
                 #todo get correct values here
                 'Ti': FreeParameter(CLDataType.from_string('double'), 'Ti', False, 0.05, 0.0, 1.0,
                                     parameter_transform=ClampTransform(), sampling_proposal=GaussianProposal(0.0001)),
                 'TR': FreeParameter(CLDataType.from_string('double'), 'TR', False, 0.05, 0.0, 1.0,
                                     parameter_transform=ClampTransform(),
                                     sampling_proposal=GaussianProposal(0.0001)),

                 'theta': FreeParameter(CLDataType.from_string('double'), 'theta', False, 1/2.0*np.pi, 0, np.pi,
                                        parameter_transform=AbsModPiTransform(),
                                        sampling_proposal=GaussianProposal(0.02),
                                        sampling_prior=AbsSinHalfPrior(),
                                        sampling_statistics=CircularGaussianPSS(),
                                        perturbation_function=lambda v: v + np.random.normal(scale=0.1, size=v.shape)),
                 'phi': FreeParameter(CLDataType.from_string('double'), 'phi', False, 1/2.0*np.pi, 0, np.pi,
                                      parameter_transform=AbsModPiTransform(),
                                      sampling_proposal=GaussianProposal(0.02),
                                      sampling_prior=AbsSinPrior(),
                                      sampling_statistics=CircularGaussianPSS(),
                                      perturbation_function=lambda v: v + np.random.normal(scale=0.1, size=v.shape)),
                 'psi': FreeParameter(CLDataType.from_string('double'), 'psi', False, 1/2.0*np.pi, 0, np.pi,
                                      parameter_transform=AbsModPiTransform(),
                                      sampling_proposal=GaussianProposal(0.02),
                                      sampling_prior=AbsSinPrior(),
                                      sampling_statistics=CircularGaussianPSS(),
                                      perturbation_function=lambda v: v + np.random.normal(scale=0.1, size=v.shape)),

                 'd': FreeParameter(
                     CLDataType.from_string('double'), 'd', False, 1.7e-9, 0, 1.0e-8,
                     parameter_transform=SinSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1e-14)),

                 'dperp0': FreeParameter(
                     CLDataType.from_string('double'), 'dperp0', False, 1.7e-10, 0, 1e-8,
                     parameter_transform=SinSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1e-15)),

                 'dperp1': FreeParameter(
                     CLDataType.from_string('double'), 'dperp1', False, 1.7e-11, 0, 1e-8,
                     parameter_transform=SinSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1e-15)),

                 'R': FreeParameter(CLDataType.from_string('double'), 'R', False, 2.0e-6, 1e-6, 20e-6,
                                    parameter_transform=CosSqrClampTransform(),
                                    sampling_proposal=GaussianProposal(1e-6)),
                 'kappa': FreeParameter(CLDataType.from_string('double'), 'kappa', False, 1, 1e-5, 2 * np.pi * 10,
                                        parameter_transform=CosSqrClampTransform(),
                                        sampling_proposal=GaussianProposal(1.0)),

                 # for use in the GDRCylinder model
                 'gamma_k': FreeParameter(CLDataType.from_string('double'), 'gamma_k', False, 1, 0, 20,
                                          parameter_transform=SinSqrClampTransform(),
                                          sampling_proposal=GaussianProposal(1.0)),
                 'gamma_beta': FreeParameter(CLDataType.from_string('double'), 'gamma_beta', False, 1, 1.0e-7, 3.0e-7,
                                             parameter_transform=SinSqrClampTransform(),
                                             sampling_proposal=GaussianProposal(1e-7)),
                 'gamma_nmr_cyl': FreeParameter(CLDataType.from_string('double'), 'gamma_nmr_cyl', True, 5, 1, 10,
                                                parameter_transform=SinSqrClampTransform(),
                                                sampling_proposal=GaussianProposal(1.0)),

                 'CLJnpZeros': ModelDataParameter(CLDataType.from_string('model_float*'), 'CLJnpZeros',
                                                  get_bessel_roots(number_of_roots=20)),
                 'CLJnpZerosLength': ModelDataParameter(CLDataType.from_string('int'), 'CLJnpZerosLength', 20),

                 # charmed default, this is the fixed parameter for the length of the GDRCylindersFixed model
                 'nmr_gamma_cyl_fixed': ModelDataParameter(CLDataType.from_string('int'), 'nmr_gamma_cyl_fixed', 6),

                 # charmed default, used in GDRCylindersFixed model
                 'gamma_radii': ModelDataParameter(
                     CLDataType.from_string('model_float*'),
                     'gamma_radii', 1e-6 * np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])),

                 # charmed default, used in GDRCylindersFixed model
                 'gamma_cyl_weights': ModelDataParameter(
                     CLDataType.from_string('model_float*'),
                     'gamma_cyl_weights', np.array([0.0211847200855742,
                                                    0.107169623942214,
                                                    0.194400551313197,
                                                    0.266676876170322,
                                                    0.214921653661151,
                                                    0.195646574827541])),
                 }
    return param_map[param_name]