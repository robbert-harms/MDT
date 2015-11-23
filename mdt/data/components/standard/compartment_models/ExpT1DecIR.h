#ifndef DMRICM_EXPT1DECIR_H
#define DMRICM_EXPT1DECIR_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the T1 Inversion Recovery model
 * @params TR the protocol value for TR
 * @params T1 the parameter T1
 */
MOT_FLOAT_TYPE cmExpT1DecIR(const MOT_FLOAT_TYPE Ti, const MOT_FLOAT_TYPE T1);

#endif // DMRICM_EXPT1DECIR_H