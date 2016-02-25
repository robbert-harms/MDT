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
MOT_FLOAT_TYPE cmTensor(const MOT_FLOAT_TYPE4 g,
                     const MOT_FLOAT_TYPE b,
                     const MOT_FLOAT_TYPE d,
                     const MOT_FLOAT_TYPE dperp,
                     const MOT_FLOAT_TYPE dperp2,
                     const MOT_FLOAT_TYPE theta,
                     const MOT_FLOAT_TYPE phi,
                     const MOT_FLOAT_TYPE psi){

    MOT_FLOAT_TYPE cos_theta;
    MOT_FLOAT_TYPE sin_theta = sincos(theta, &cos_theta);
    MOT_FLOAT_TYPE cos_phi;
    MOT_FLOAT_TYPE sin_phi = sincos(phi, &cos_phi);
    MOT_FLOAT_TYPE cos_psi;
    MOT_FLOAT_TYPE sin_psi = sincos(psi, &cos_psi);

    MOT_FLOAT_TYPE4 n1 = (MOT_FLOAT_TYPE4)(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta, 0.0);

    // rotate around n1
    // this code is optimized for memory consumption. View the git history for human readable previous versions.
    MOT_FLOAT_TYPE tmp = sin(theta+(M_PI_2)); // using tmp as the rotation factor (90 degrees)
    MOT_FLOAT_TYPE4 n2 = (MOT_FLOAT_TYPE4)(tmp * cos_phi, tmp * sin_phi, cos(theta+(M_PI_2)), 0.0);
    tmp = select(1, -1, n1.z < 0 || ((n1.z == 0.0) && n1.x < 0.0)); // using tmp as the multiplier
    n2 = n2 * cos_psi + (cross(n2, tmp * n1) * sin_psi) + (tmp * n1 * dot(tmp * n1, n2) * (1-cos_psi));

    return exp(-b * (d *      pown(dot(n1, g), 2) +
                     dperp *  pown(dot(n2, g), 2) +
                     dperp2 * pown(dot(cross(n1, n2), g), 2)
                  )
               );
}
