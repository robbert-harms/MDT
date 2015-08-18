#ifndef DMRICM_EXPT2DEC_CL
#define DMRICM_EXPT2DEC_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the T2 model
 * @params TE the protocol value for TE
 * @params T2 the parameter T2
 */
double cmExpT2Dec(const double TE, const double T2){
    return exp(-TE / T2);
}

#endif // DMRICM_EXPT2DEC_CL