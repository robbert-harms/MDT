#ifndef DMRICM_CHARMEDRESTRICTED_H
#define DMRICM_CHARMEDRESTRICTED_H

/**
 * Author = Robbert Harms
 * Date = 2014-11-06
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmCharmedRestricted(const model_float4 g,
                                const model_float b,
                                const model_float GAMMA2_G2_delta2,
                                const model_float TE,
                                const model_float d,
                                const model_float theta,
                                const model_float phi);

#endif // DMRICM_CHARMEDRESTRICTED_H