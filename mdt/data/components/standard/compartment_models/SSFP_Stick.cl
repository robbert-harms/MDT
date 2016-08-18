/**
 * Author = Francisco L. Fritz
 * Date = 2016-18-08
 * License = LGPL v3
 * Maintainer = Francisco L. Fritz
 * Email = francisco.lagos@maastrichtuniversity.nl
 */

/**
 * Following Buxton equation and Miller paper (2008), the equation receives the following variables:
 * T1 and T2 values, TR, Gradient amplitude g, Gradient duration d, flip angle alpha, B1 map and Diffusion D.
 */

mot_float_type cmSSFP_Stick(const mot_float_type4 g,
                            const mot_float_type d,
                            const mot_float_type TR,
                            const mot_float_type flip_angle,
                            const mot_float_type B1map,
                            const mot_float_type T1map,
                            const mot_float_type T2map){

    mot_float_type E1_var = exp(- TR / T1map )

    mot_float_type E2_var = exp(- TR / T2map )

    mot_float_type A1_var = exp(- pow(gamma*g*d,2) * TR * D)

    mot_float_type A2_var = exp(- pow(gamma*g*d,2) * d * D)

    mot_float_type s_var = E2_var * A1_var * pow(A2_var,-4/3) * (1 - E1_var * cos(flip_angle*B1map)) + E2_var * pow(A2_var,-1/3) * (cos(flip_angle*B1map) - 1)

    mot_float_type r_var = 1 - E1_var * cos(flip_angle*B1map) + pow(E2_var,2) * A1_var * pow(A2_var,1/3) * (cos(flip_angle*B1map) - E1_var)

    mot_float_type K_var = (1 - E1_var * A1_var * cos(flip_angle*B1map) + pow(E2_var,2) * pow(A1_var,2) * pow(A2_var,-2/3) * (cos(flip_angle*B1map) - E1_var * A1_var))/(E2_var * A1_var * pow(A2_var,-4/3) * (1 + cos(flip_angle*B1map)) * (1 - E1_var * A1_var))

    mot_float_type F1_var = K_var - sqrt(pow(K_var,2) - pow(A2_var,2))

    mot_float_type signal = - ((1 - E1_var)*E2_var*pow(A2_var,-2/3)*(F1_var - E2_var * A1_var * pow(A2_var,2/3))*sin(flip_angle*B1map))/(r_var - F1_var)

    return (mot_float_type)signal;
}
