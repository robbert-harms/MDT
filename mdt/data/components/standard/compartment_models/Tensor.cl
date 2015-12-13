/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

MOT_FLOAT_TYPE4 Tensor_rotateVector(const MOT_FLOAT_TYPE4 vector, const MOT_FLOAT_TYPE4 axis_rotate, const MOT_FLOAT_TYPE psi);

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

    MOT_FLOAT_TYPE sinT = sin(theta);
    MOT_FLOAT_TYPE sinP = sin(phi);
    MOT_FLOAT_TYPE cosP = cos(phi);
    MOT_FLOAT_TYPE rst = sin(theta+(M_PI_2));

    MOT_FLOAT_TYPE4 n1 = (MOT_FLOAT_TYPE4)(cosP * sinT, sinP * sinT, cos(theta), 0.0);
    MOT_FLOAT_TYPE4 n2 = Tensor_rotateVector((MOT_FLOAT_TYPE4)(rst * cosP, rst * sinP, cos(theta+(M_PI_2)), 0.0), n1, psi);

    return exp(-b * (d *      pown(dot(n1, g), 2) +
                     dperp *  pown(dot(n2, g), 2) +
                     dperp2 * pown(dot(cross(n1, n2), g), 2)
                  )
               );

}

MOT_FLOAT_TYPE4 Tensor_rotateVector(const MOT_FLOAT_TYPE4 vector, const MOT_FLOAT_TYPE4 axis_rotate, const MOT_FLOAT_TYPE psi){
    MOT_FLOAT_TYPE4 n1 = axis_rotate;
    if(axis_rotate.z < 0 || ((axis_rotate.z == 0.0) && (axis_rotate.x < 0.0))){
    	n1 *= -1;
    }
    return vector * cos(psi) + (cross(vector, n1) * sin(psi)) + (n1 * dot(n1, vector) * (1-cos(psi)));
}

