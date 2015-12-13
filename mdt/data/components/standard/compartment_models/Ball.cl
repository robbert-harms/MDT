/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the Ball model.
 * @params b the scheme value for b
 * @params d the parameter d
 */
MOT_FLOAT_TYPE cmBall(const MOT_FLOAT_TYPE b, const MOT_FLOAT_TYPE d){
    return exp(-d * b);
}

