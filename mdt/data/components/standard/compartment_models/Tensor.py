from mdt.component_templates.compartment_models import CompartmentTemplate
from mdt.post_processing import DTIMeasures

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Tensor(CompartmentTemplate):

    parameter_list = ('g', 'b', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi')
    dependency_list = ['TensorApparentDiffusion']
    cl_code = '''
        mot_float_type adc = TensorApparentDiffusion(theta, phi, psi, d, dperp0, dperp1, g);
        return exp(-b * adc);
    '''
    extra_prior = 'return dperp1 < dperp0 && dperp0 < d;'
    post_optimization_modifiers = [DTIMeasures.post_optimization_modifier]
    extra_optimization_maps = [DTIMeasures.extra_optimization_maps]
    extra_sampling_maps = [DTIMeasures.extra_sampling_maps]
