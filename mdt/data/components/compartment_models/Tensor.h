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
model_float cmTensor(const model_float4 g,
                     const model_float b,
                     const model_float d,
                     const model_float dperp,
                     const model_float dperp2,
                     const model_float theta,
                     const model_float phi,
                     const model_float psi);

#endif // DMRICM_TENSOR_H