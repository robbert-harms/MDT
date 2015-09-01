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
                const double d,
                const double dperp,
                const double dperp2,
                const double theta,
                const double phi,
                const double psi);

#endif // DMRICM_TENSOR_H