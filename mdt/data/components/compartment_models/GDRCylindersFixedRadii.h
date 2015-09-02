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
model_float cmGDRCylindersFixedRadii(const model_float4 g,
                                     const model_float G,
                                     const model_float Delta,
                                     const model_float delta,
                                     const model_float d,
                                     const model_float theta,
                                     const model_float phi,
                                     global const model_float* const gamma_cyl_radii,
                                     global const model_float* const gamma_cyl_weights,
                                     const int nmr_gamma_cyl_fixed,
                                     global const model_float* const CLJnpZeros,
                                     const int CLJnpZerosLength);

#endif // DMRICM_GDRCYLINDERSFIXEDRADII_H