#ifndef DMRICM_ASTROCYLINDERS_H
#define DMRICM_ASTROCYLINDERS_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmAstroCylinders(const model_float4 g,
                        const model_float b,
                        const model_float G,
                        const model_float Delta,
                        const model_float delta,
                        const double d,
                        const double R,
                        global const model_float* const CLJnpZeros,
                        const int CLJnpZerosLength);

#endif // DMRICM_ASTROCYLINDERS_H