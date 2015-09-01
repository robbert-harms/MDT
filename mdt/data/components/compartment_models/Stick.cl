#ifndef DMRICM_STICK_CL
#define DMRICM_STICK_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */


/**
 * Generate the compartment model signal for the Stick model.
 * @params g the protocol gradient vector with (x, y, z)
 * @params b the protocol b
 * @params d the parameter d
 * @params theta the parameter theta
 * @params phi the parameter phi
 */
model_float cmStick(const model_float4 g,
               const model_float b,
               const double d,
               const double theta,
               const double phi){
    return exp(-b * d * pown(dot(g, (model_float4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0)), 2));
}

#endif // DMRICM_STICK_CL