#ifndef DMRICM_ASTROCYLINDERS_H
#define DMRICM_ASTROCYLINDERS_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

MOT_FLOAT_TYPE cmAstroCylinders(const MOT_FLOAT_TYPE4 g,
                             const MOT_FLOAT_TYPE b,
                             const MOT_FLOAT_TYPE G,
                             const MOT_FLOAT_TYPE Delta,
                             const MOT_FLOAT_TYPE delta,
                             const MOT_FLOAT_TYPE d,
                             const MOT_FLOAT_TYPE R,
                             global const MOT_FLOAT_TYPE* const CLJnpZeros,
                             const int CLJnpZerosLength);

#endif // DMRICM_ASTROCYLINDERS_H