from mdt.components_loader import component_import


class SSFP_Tensor(component_import('standard.compartment_models.Tensor', 'Tensor')):

    parameter_list = ('g', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi', 'delta',
                      'G', 'TR', 'flip_angle', 'b1_static', 'T1_static', 'T2_static')
    dependency_list = ('SSFP', 'TensorSphericalToCartesian')
    cl_code = '''
        mot_float_type4 vec0, vec1, vec2;
        TensorSphericalToCartesian(theta, phi, psi, &vec0, &vec1, &vec2);

        mot_float_type adc = (d *      pown(dot(vec0, g), 2) +
                              dperp0 * pown(dot(vec1, g), 2) +
                              dperp1 * pown(dot(vec2, g), 2));

        return SSFP(adc, delta, G, TR, flip_angle, b1_static, T1_static, T2_static);
    '''
