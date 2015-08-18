#ifndef DMRICM_GDRCYLINDERSFIXEDRADII_H
#define DMRICM_GDRCYLINDERSFIXEDRADII_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the Gamma Distributed Radii model.
 *
 * This is a fixed version of the GDRCylinders model. This means that the different radii are not calculated
 * dynamically by means of a Gamma distribution. Rather, the list of radii and the corresponding weights
 * are given as fixed values.
 *
 * @params gamma_cyl_radii, the list of radii that should be used for calculating the cylinders.
 * @params gamma_cyl_weights, the list of weights per radius.
 * @params nmr_gamma_cyl, the number of cylinders we provided
 *
 */
double cmGDRCylindersFixedRadii(const double4 g,
                                const double G,
                                const double Delta,
                                const double delta,
                                const double d,
                                const double theta,
                                const double phi,
                                global const double* const gamma_cyl_radii,
                                global const double* const gamma_cyl_weights,
                                const int nmr_gamma_cyl_fixed,
                                global const double* const CLJnpZeros,
                                const int CLJnpZerosLength);

#endif // DMRICM_GDRCYLINDERSFIXEDRADII_H