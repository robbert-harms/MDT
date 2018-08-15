from mdt import CompartmentTemplate
from mdt.lib.post_processing import DTIMeasures, noddi_dti_maps

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Tensor(CompartmentTemplate):

    parameters = ('g', 'b', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi')
    dependencies = ['TensorApparentDiffusion']
    cl_code = '''
        mot_float_type adc = TensorApparentDiffusion(theta, phi, psi, d, dperp0, dperp1, g);
        return exp(-b * adc);
    '''
    extra_prior = 'return dperp1 < dperp0 && dperp0 < d;'
    post_optimization_modifiers = [DTIMeasures.post_optimization_modifier]
    extra_optimization_maps = [
        DTIMeasures.extra_optimization_maps,
        noddi_dti_maps
    ]
    extra_sampling_maps = [DTIMeasures.extra_sampling_maps]
