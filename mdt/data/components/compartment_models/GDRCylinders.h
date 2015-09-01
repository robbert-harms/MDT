#ifndef DMRICM_GDRCYLINDERS_H
#define DMRICM_GDRCYLINDERS_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Gamma distributed radii cylinders.
 *
 * This model sums a number of CylinderGPD models with the radius of each cylinder created
 * by using a gamma distribution with as parameters gamma_k and gamma_beta.
 *
 * @param gamma_k the gamma shape parameter
 * @param gamma_beta the gamma scale parameter
 * @param gamma_nmr_cyl the number of different cylinders we calculate
 */
model_float cmGDRCylinders(const double4 g,
                      const double G,
                      const double Delta,
                      const double delta,
                      const double d,
                      const double theta,
                      const double phi,
                      const double gamma_k,
                      const double gamma_beta,
                      const double gamma_nmr_cyl,
                      global const double* const CLJnpZeros,
                      const int CLJnpZerosLength);

#endif // DMRICM_GDRCYLINDERS_H