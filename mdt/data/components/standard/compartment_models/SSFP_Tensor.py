from mdt.components_loader import component_import


class SSFP_Tensor(component_import('standard.compartment_models.Tensor', 'Tensor')):

    parameter_list = ('g', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi', 'delta',
                      'G', 'TR', 'flip_angle', 'b1_static', 'T1_static', 'T2_static')
    dependency_list = ('SSFP', 'TensorApparentDiffusion')
    cl_code = '''
        mot_float_type adc = TensorApparentDiffusion(theta, phi, psi, d, dperp0, dperp1, g);
        return SSFP(adc, delta, G, TR, flip_angle, b1_static, T1_static, T2_static);
    '''
