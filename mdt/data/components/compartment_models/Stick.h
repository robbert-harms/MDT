#ifndef DMRICM_STICK_H
#define DMRICM_STICK_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */


/**
 * Generate the compartment model signal for the Stick model.
 * @params g the protocol gradient vector with (x, y, z, 0.0)
 * @params b the protocol b
 * @params d the parameter d
 * @params theta the parameter theta
 * @params phi the parameter phi
 */
model_float cmStick(const double4 g,
               const double b,
               const double d,
               const double theta,
               const double phi);

#endif // DMRICM_STICK_H