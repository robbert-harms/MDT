#ifndef DMRICM_TENSOR_CL
#define DMRICM_TENSOR_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

double4 Tensor_rotateVector(const double4 vector, const double4 axis_rotate, const double psi);

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
double cmTensor(const double4 g,
                const double b,
                const double d,
                const double dperp,
                const double dperp2,
                const double theta,
                const double phi,
                const double psi){

    double sinT = sin(theta);
    double sinP = sin(phi);
    double cosP = cos(phi);
    double rst = sin(theta+(M_PI_2));

    double4 n1 = (double4)(cosP * sinT, sinP * sinT, cos(theta), 0.0);
    double4 n2 = Tensor_rotateVector((double4)(rst * cosP, rst * sinP, cos(theta+(M_PI_2)), 0.0), n1, psi);

    return exp(-b * (d *      pown(dot(n1, g), 2) +
                     dperp *  pown(dot(n2, g), 2) +
                     dperp2 * pown(dot(cross(n1, n2), g), 2)
                  )
               );

}

double4 Tensor_rotateVector(const double4 vector, const double4 axis_rotate, const double psi){
    double4 n1 = axis_rotate;
    if(axis_rotate.z < 0 || ((axis_rotate.z == 0.0) && (axis_rotate.x < 0.0))){
    	n1 *= -1;
    }
    return vector * cos(psi) + (cross(vector, n1) * sin(psi)) + (n1 * dot(n1, vector) * (1-cos(psi)));
}

#endif // DMRICM_TENSOR_CL