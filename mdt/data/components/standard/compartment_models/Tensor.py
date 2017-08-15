import numpy as np
from mdt.component_templates.compartment_models import CompartmentTemplate
from mdt.cl_routines.mapping.dti_measures import DTIMeasures
from mdt.utils import tensor_spherical_to_cartesian

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_dti_measures_modifier():
    measures_calculator = DTIMeasures()
    return_names = measures_calculator.get_output_names()

    def modifier_routine(results_dict):
        measures = measures_calculator.calculate(results_dict)
        return [measures[name] for name in return_names]

    return return_names, modifier_routine


def extra_covariance_samples(theta, phi, psi):
    return np.rollaxis(np.concatenate(tensor_spherical_to_cartesian(theta, phi, psi), axis=2), 2, 1)


class Tensor(CompartmentTemplate):

    parameter_list = ('g', 'b', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi')
    dependency_list = ['TensorApparentDiffusion']
    cl_code = '''
        mot_float_type adc = TensorApparentDiffusion(theta, phi, psi, d, dperp0, dperp1, g);
        return exp(-b * adc);
    '''
    extra_prior = 'return dperp1 < dperp0 && dperp0 < d;'

    auto_add_cartesian_vector = False
    post_optimization_modifiers = [get_dti_measures_modifier()]

    auto_sampling_covar_cartesian = False
    sampling_covar_extras = [(('theta', 'phi', 'psi'), ('vec0_x', 'vec0_y', 'vec0_z',
                                                        'vec1_x', 'vec1_y', 'vec1_z',
                                                        'vec2_x', 'vec2_y', 'vec2_z'), extra_covariance_samples)]
    sampling_covar_exclude = ['theta', 'phi', 'psi']
