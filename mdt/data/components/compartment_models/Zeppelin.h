#ifndef DMRICM_ZEPPELIN_H
#define DMRICM_ZEPPELIN_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-26
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

 /**
 * Generate the compartment model signal for the Zeppelin model.
 * @params g the protocol gradient vector with (x, y, z)
 * @params b the protocol b
 * @params d the parameter d
 * @params theta the parameter theta
 * @params phi the parameter phi
 * @params dperp the perpendicular diffusivity
 */
model_float cmZeppelin(const double4 g,
                  const double b,
                  const double d,
                  const double dperp,
                  const double theta,
                  const double phi);

#endif // DMRICM_ZEPPELIN_H