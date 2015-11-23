#ifndef DMRICM_TENSOR_H
#define DMRICM_TENSOR_H

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
                     const MOT_FLOAT_TYPE psi);

#endif // DMRICM_TENSOR_H