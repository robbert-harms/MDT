from mdt.components_loader import component_import


class SSFP_Tensor(component_import('standard.compartment_models.Tensor', 'Tensor')):

    parameter_list = ('g', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi', 'delta',
                      'G', 'TR', 'flip_angle', 'b1_static', 'T1_exvivo', 'T2_exvivo')
    dependency_list = ('SSFP',)
    cl_code = '''
        mot_float_type cos_theta;
        mot_float_type sin_theta = sincos(theta, &cos_theta);
        mot_float_type cos_phi;
        mot_float_type sin_phi = sincos(phi, &cos_phi);
        mot_float_type cos_psi;
        mot_float_type sin_psi = sincos(psi, &cos_psi);

        mot_float_type4 n1 = (mot_float_type4)(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta, 0.0);

        // rotate n1 by 90 degrees, changing, x, y and z
        mot_float_type rotation_factor = sin(theta+(M_PI_2_F));
        mot_float_type4 n2 = (mot_float_type4)(rotation_factor * cos_phi,
                                               rotation_factor * sin_phi,
                                               cos(theta+(M_PI_2_F)),
                                               0.0);

        // uses Rodrigues' formula to rotate n2 by psi around n1
        // using a multiplication factor "select(1, -1, n1.z < 0 || ((n1.z == 0.0) && n1.x < 0.0))" to
        // prevent commutative problems in the cross product between n1xn2
        n2 = n2 * cos_psi
                    + (cross(n2, select(1, -1, n1.z < 0 || ((n1.z == 0.0) && n1.x < 0.0)) * n1) * sin_psi)
                    + (n1 * dot(n1, n2) * (1-cos_psi));

        mot_float_type adc = (d *      pown(dot(n1, g), 2) +
                              dperp0 * pown(dot(n2, g), 2) +
                              dperp1 * pown(dot(cross(n1, n2), g), 2)
                              );

        return SSFP(adc, delta, G, TR, flip_angle, b1_static, T1_exvivo, T2_exvivo);
    '''