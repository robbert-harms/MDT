#ifndef DMRICM_EXPT1DECIR_CL
#define DMRICM_EXPT1DECIR_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the T1 Inversion Recovery model
 * @params Ti the protocol value for Ti (time inversion recovery)
 * @params T1 the parameter T1
 */
MOT_FLOAT_TYPE cmExpT1DecIR(const MOT_FLOAT_TYPE Ti, const MOT_FLOAT_TYPE T1){
    return abs(1 - 2 * exp(-Ti / T1));
}

#endif // DMRICM_EXPT1DECIR_CL