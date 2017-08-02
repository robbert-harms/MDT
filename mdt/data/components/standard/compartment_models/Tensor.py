import numpy as np
from mdt.component_templates.compartment_models import CompartmentTemplate
from mdt.cl_routines.mapping.dti_measures import DTIMeasures
from mdt.utils import eigen_vectors_from_tensor

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_dti_measures_modifier():
    measures_calculator = DTIMeasures()
    return_names = measures_calculator.get_output_names()

    def modifier_routine(results_dict):
        eigen_vectors = eigen_vectors_from_tensor(results_dict['theta'], results_dict['phi'], results_dict['psi'])

        eigen_values = np.atleast_2d(np.squeeze(np.dstack([results_dict['d'],
                                                           results_dict['dperp0'],
                                                           results_dict['dperp1']])))

        measures = measures_calculator.calculate(eigen_values, eigen_vectors)
        return [measures[name] for name in return_names]

    return return_names, modifier_routine


class Tensor(CompartmentTemplate):

    parameter_list = ('g', 'b', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi')
    dependency_list = ['TensorApparentDiffusion']
    cl_code = '''
        mot_float_type adc = TensorApparentDiffusion(theta, phi, psi, d, dperp0, dperp1, g);
        return exp(-b * adc);
    '''
    prior = 'return dperp1 < dperp0 && dperp0 < d;'
    auto_add_cartesian_vector = False
    post_optimization_modifiers = [get_dti_measures_modifier()]
