from mdt.component_templates.compartment_models import CompartmentTemplate


class SSFP_Zeppelin(CompartmentTemplate):

    parameter_list = ('g', 'd', 'dperp0', 'theta', 'phi', 'delta', 'G', 'TR', 'flip_angle', 'b1_static', 'T1', 'T2')
    dependency_list = ('SSFP',)
    cl_code = '''
        mot_float_type adc = dperp0 + ((d - dperp0) * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                                   sin(phi) * sin(theta), cos(theta), 0.0)), 2);

        return SSFP(adc, delta, G, TR, flip_angle, b1_static, T1, T2);
    '''
