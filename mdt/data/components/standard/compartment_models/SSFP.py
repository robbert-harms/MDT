from mdt import CompartmentTemplate
from mdt.lib.post_processing import DTIMeasures

__author__ = 'Robbert Harms'
__date__ = '2017-08-03'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class SSFP_Ball(CompartmentTemplate):

    parameters = ('d', 'delta', 'G', 'TR', 'flip_angle', 'b1', 'T1', 'T2')
    dependencies = ('SSFP',)
    cl_code = '''
        return SSFP(d, delta, G, TR, flip_angle, b1, T1, T2);
    '''


class SSFP_Stick(CompartmentTemplate):

    parameters = ('g', 'd', 'theta', 'phi', 'delta', 'G', 'TR', 'flip_angle', 'b1', 'T1', 'T2')
    dependencies = ('SSFP',)
    cl_code = '''
        mot_float_type adc = d * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta), sin(phi) * sin(theta),
                                                               cos(theta), 0.0)), 2);

        return SSFP(adc, delta, G, TR, flip_angle, b1, T1, T2);
    '''


class SSFP_Tensor(CompartmentTemplate):

    parameters = ('g', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi', 'delta',
                  'G', 'TR', 'flip_angle', 'b1', 'T1', 'T2')
    dependencies = ('SSFP', 'TensorApparentDiffusion')
    cl_code = '''
        mot_float_type adc = TensorApparentDiffusion(theta, phi, psi, d, dperp0, dperp1, g);
        return SSFP(adc, delta, G, TR, flip_angle, b1, T1, T2);
    '''
    extra_prior = 'return dperp1 < dperp0 && dperp0 < d;'
    post_optimization_modifiers = [DTIMeasures.post_optimization_modifier]
    extra_optimization_maps = [DTIMeasures.extra_optimization_maps]
    extra_sampling_maps = [DTIMeasures.extra_sampling_maps]


class SSFP_Zeppelin(CompartmentTemplate):

    parameters = ('g', 'd', 'dperp0', 'theta', 'phi', 'delta', 'G', 'TR', 'flip_angle', 'b1', 'T1', 'T2')
    dependencies = ('SSFP',)
    cl_code = '''
        mot_float_type adc = dperp0 + ((d - dperp0) * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                                   sin(phi) * sin(theta), cos(theta), 0.0)), 2);

        return SSFP(adc, delta, G, TR, flip_angle, b1, T1, T2);
    '''
