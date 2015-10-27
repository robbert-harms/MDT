#ifndef DMRICM_EXPT1DECTM_CL
#define DMRICM_EXPT1DECTM_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmExpT1DecTM(const model_float TM, const model_float T1){
    return exp(-TM / T1);
}

#endif // DMRICM_EXPT1DECTM_CL