#ifndef DMRICM_SPHEREGPD_H
#define DMRICM_SPHEREGPD_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmSphereGPD(const double Delta,
                   const double delta,
                   const double d,
                   const double R,
                   global const double* const CLJnpZeros,
                   const int CLJnpZerosLength);

#endif // DMRICM_SPHEREGPD_H