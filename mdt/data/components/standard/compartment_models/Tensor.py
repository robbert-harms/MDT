from mdt.component_templates.compartment_models import CompartmentTemplate
from mdt.cl_routines.mapping.dti_measures import DTIMeasures

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
