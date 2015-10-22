#ifndef DMRICM_EXPT1DECTM_H
#define DMRICM_EXPT1DECTM_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the TM STEAM STE signal model
 * @params TM the protocol value for TM
 * @params T1 the parameter T1
 */
model_float cmExpT1DecTM(const model_float TM, const model_float T1);

#endif // DMRICM_EXPT1DECTM_H