#ifndef DMRICM_ZEPPELIN_CL
#define DMRICM_ZEPPELIN_CL

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
double cmZeppelin(const double4 g,
                  const double b,
                  const double d,
                  const double dperp,
                  const double theta,
                  const double phi){
    return exp(-b * (
                ((d - dperp) *
                 pown(dot(g, (double4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0)), 2))
                  + dperp));
}

#endif // DMRICM_ZEPPELIN_CL