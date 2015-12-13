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
MOT_FLOAT_TYPE cmStick(const MOT_FLOAT_TYPE4 g,
                    const MOT_FLOAT_TYPE b,
                    const MOT_FLOAT_TYPE d,
                    const MOT_FLOAT_TYPE theta,
                    const MOT_FLOAT_TYPE phi){
    return exp(-b * d * pown(dot(g, (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0)), 2));
}

