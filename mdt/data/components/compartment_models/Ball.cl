#ifndef DMRICM_BALL_CL
#define DMRICM_BALL_CL

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
model_float cmBall(const model_float b, const double d){
    return exp(-d * b);
}

#endif // DMRICM_BALL_CL