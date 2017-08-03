from mdt.component_templates.compartment_models import CompartmentTemplate
from mdt.components_loader import component_import

__author__ = 'Robbert Harms'
__date__ = '2017-08-03'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class SSFP_Ball(CompartmentTemplate):

    parameter_list = ('d', 'delta', 'G', 'TR', 'flip_angle', 'b1_static', 'T1_static', 'T2_static')
    dependency_list = ('SSFP',)
    cl_code = '''
        return SSFP(d, delta, G, TR, flip_angle, b1_static, T1_static, T2_static);
    '''


class SSFP_Stick(CompartmentTemplate):

    parameter_list = ('g', 'd', 'theta', 'phi', 'delta', 'G', 'TR', 'flip_angle', 'b1_static', 'T1_static', 'T2_static')
    dependency_list = ('SSFP',)
    cl_code = '''
        mot_float_type adc = d * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta), sin(phi) * sin(theta),
                                                               cos(theta), 0.0)), 2);

        return SSFP(adc, delta, G, TR, flip_angle, b1_static, T1_static, T2_static);
    '''


class SSFP_Tensor(component_import('standard.compartment_models.Tensor', 'Tensor')):

    parameter_list = ('g', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi', 'delta',
                      'G', 'TR', 'flip_angle', 'b1_static', 'T1_static', 'T2_static')
    dependency_list = ('SSFP', 'TensorApparentDiffusion')
    cl_code = '''
        mot_float_type adc = TensorApparentDiffusion(theta, phi, psi, d, dperp0, dperp1, g);
        return SSFP(adc, delta, G, TR, flip_angle, b1_static, T1_static, T2_static);
    '''


class SSFP_Zeppelin(CompartmentTemplate):

    parameter_list = ('g', 'd', 'dperp0', 'theta', 'phi', 'delta', 'G', 'TR', 'flip_angle', 'b1_static', 'T1', 'T2')
    dependency_list = ('SSFP',)
    cl_code = '''
        mot_float_type adc = dperp0 + ((d - dperp0) * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                                   sin(phi) * sin(theta), cos(theta), 0.0)), 2);

        return SSFP(adc, delta, G, TR, flip_angle, b1_static, T1, T2);
    '''
