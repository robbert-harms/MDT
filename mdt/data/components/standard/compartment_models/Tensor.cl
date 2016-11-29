/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the Tensor model.
 * @params g the protocol gradient vector with (x, y, z)
 * @params b the protocol b
 * @params d the parameter d
 * @params theta the parameter theta
 * @params phi the parameter phi
 * @params dperp parameter perpendicular diffusion 1
 * @params dperp2 parameter perpendicular diffusion 2
 * @params psi the third rotation angle
 */
mot_float_type cmTensor(
        const mot_float_type4 g,
        const mot_float_type b,
        const mot_float_type d,
        const mot_float_type dperp,
        const mot_float_type dperp2,
        const mot_float_type theta,
        const mot_float_type phi,
        const mot_float_type psi){

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

    return exp(-b * (d *      pown(dot(n1, g), 2) +
                     dperp *  pown(dot(n2, g), 2) +
                     dperp2 * pown(dot(cross(n1, n2), g), 2)
                  )
               );
}
