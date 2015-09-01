#ifndef DMRICM_CYLINDERGPD_H
#define DMRICM_CYLINDERGPD_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the CylinderGPD model.
 */
model_float cmCylinderGPD(const model_float4 g,
                     const model_float G,
                     const model_float Delta,
                     const model_float delta,
                     const double d,
                     const double theta,
                     const double phi,
                     const double R,
                     global const model_float* const CLJnpZeros,
                     const int CLJnpZerosLength);

#endif // DMRICM_CYLINDERGPD_H