#ifndef DMRICM_SPHEREGPD_H
#define DMRICM_SPHEREGPD_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmSphereGPD(const model_float Delta,
                        const model_float delta,
                        const model_float d,
                        const model_float R,
                        global const model_float* const CLJnpZeros,
                        const int CLJnpZerosLength);

#endif // DMRICM_SPHEREGPD_H