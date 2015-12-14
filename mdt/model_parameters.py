import numpy as np

from mot.base import DataType, ProtocolParameter, FreeParameter, ModelDataParameter
from mot.model_building.parameter_functions.priors import AbsSinPrior, AbsSinHalfPrior
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mot.model_building.parameter_functions.sample_statistics import CircularGaussianPSS
from mot.model_building.parameter_functions.transformations import ClampTransform, \
    AbsModPiTransform, SinSqrClampTransform, CosSqrClampTransform

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
    param_map = {'g': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE4'), 'g'),
                 'G': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'G'),
                 'Delta': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'Delta'),
                 'delta': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'delta'),
                 'b': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'b'),
                 'q': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'q'),

                 'GAMMA2_G2_delta2': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'),
                                                               'GAMMA2_G2_delta2'),
                 'TE': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'TE'),
                 'TM': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'TM'),
                 'Ti': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'Ti'),
                 'TR': ProtocolParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'TR'),

                 'T1': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'T1', False, 0.03, 0.0, 4.0,
                     parameter_transform=ClampTransform(),
                     sampling_proposal=GaussianProposal(0.0001)),
                 'T2': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'T2', False, 0.01, 0.0, 2.0,
                     parameter_transform=ClampTransform(),
                     sampling_proposal=GaussianProposal(0.0001)),

                 'theta': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'theta', False, 1 / 2.0 * np.pi,
                     0, np.pi,
                     parameter_transform=AbsModPiTransform(),
                     sampling_proposal=GaussianProposal(0.02),
                     sampling_prior=AbsSinHalfPrior(),
                     sampling_statistics=CircularGaussianPSS(),
                     perturbation_function=lambda v: v + np.random.normal(scale=0.1, size=v.shape)),
                 'phi': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'phi', False, 1 / 2.0 * np.pi, 0, np.pi,
                     parameter_transform=AbsModPiTransform(),
                     sampling_proposal=GaussianProposal(0.02),
                     sampling_prior=AbsSinPrior(),
                     sampling_statistics=CircularGaussianPSS(),
                     perturbation_function=lambda v: v + np.random.normal(scale=0.1, size=v.shape)),
                 'psi': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'psi', False, 1 / 2.0 * np.pi, 0, np.pi,
                     parameter_transform=AbsModPiTransform(),
                     sampling_proposal=GaussianProposal(0.02),
                     sampling_prior=AbsSinPrior(),
                     sampling_statistics=CircularGaussianPSS(),
                     perturbation_function=lambda v: v + np.random.normal(scale=0.1, size=v.shape)),

                 'd': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'd', False, 1.7e-9, 0, 1.0e-8,
                     parameter_transform=SinSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1e-14)),

                 'dperp0': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'dperp0', False, 1.7e-10, 0, 1e-8,
                     parameter_transform=SinSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1e-15)),

                 'dperp1': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'dperp1', False, 1.7e-11, 0, 1e-8,
                     parameter_transform=SinSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1e-15)),

                 'R': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'R', False, 2.0e-6, 1e-6, 20e-6,
                     parameter_transform=CosSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1e-6)),
                 'kappa': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'kappa', False, 1, 1e-5, 2 * np.pi,
                     parameter_transform=CosSqrClampTransform(),
                     sampling_proposal=GaussianProposal(0.1)),

                 # for use in the GDRCylinder model
                 'gamma_k': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'gamma_k', False, 1, 0, 20,
                     parameter_transform=SinSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1.0)),
                 'gamma_beta': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'gamma_beta', False, 1, 1.0e-7,
                     3.0e-7,
                     parameter_transform=SinSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1e-7)),
                 'gamma_nmr_cyl': FreeParameter(
                     DataType.from_string('MOT_FLOAT_TYPE'), 'gamma_nmr_cyl', True, 5, 1, 10,
                     parameter_transform=SinSqrClampTransform(),
                     sampling_proposal=GaussianProposal(1.0)),

                 # charmed default, used in GDRCylindersFixed model
                 'gamma_radii': ModelDataParameter(
                     DataType.from_string('MOT_FLOAT_TYPE*').set_address_space_qualifier('global')
                                                            .set_pre_data_type_type_qualifiers(['const'])
                                                            .set_post_data_type_type_qualifier('const'),
                     'gamma_radii', 1e-6 * np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])),

                 # charmed default, used in GDRCylindersFixed model
                 'gamma_cyl_weights': ModelDataParameter(
                     DataType.from_string('MOT_FLOAT_TYPE*').set_address_space_qualifier('global')
                                                            .set_pre_data_type_type_qualifiers(['const'])
                                                            .set_post_data_type_type_qualifier('const'),
                     'gamma_cyl_weights', np.array([0.0211847200855742,
                                                    0.107169623942214,
                                                    0.194400551313197,
                                                    0.266676876170322,
                                                    0.214921653661151,
                                                    0.195646574827541])),

                 'nmr_gamma_cyl_weights': ModelDataParameter(DataType.from_string('int'), 'nmr_gamma_cyl_weights', 6)

                 }
    return param_map[param_name]
