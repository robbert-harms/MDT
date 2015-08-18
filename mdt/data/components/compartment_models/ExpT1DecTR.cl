#ifndef DMRICM_EXPT1DECTR_CL
#define DMRICM_EXPT1DECTR_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the T1 model for TR recovery
 * @params TR the protocol value for TR
 * @params T1 the parameter T1
 */
double cmExpT1DecTR(const double TR, const double T1){
    return abs(1 - exp(-TR / T1));
}

#endif // DMRICM_EXPT1DECTR_CL