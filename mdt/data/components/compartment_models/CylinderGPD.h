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
double cmCylinderGPD(const double4 g,
                     const double G,
                     const double Delta,
                     const double delta,
                     const double d,
                     const double theta,
                     const double phi,
                     const double R,
                     global const double* const CLJnpZeros,
                     const int CLJnpZerosLength);

#endif // DMRICM_CYLINDERGPD_H