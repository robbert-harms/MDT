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
MOT_FLOAT_TYPE cmGDRCylindersFixedRadii(const MOT_FLOAT_TYPE4 g,
                                     const MOT_FLOAT_TYPE G,
                                     const MOT_FLOAT_TYPE Delta,
                                     const MOT_FLOAT_TYPE delta,
                                     const MOT_FLOAT_TYPE d,
                                     const MOT_FLOAT_TYPE theta,
                                     const MOT_FLOAT_TYPE phi,
                                     global const MOT_FLOAT_TYPE* const gamma_cyl_radii,
                                     global const MOT_FLOAT_TYPE* const gamma_cyl_weights,
                                     const int nmr_gamma_cyl_fixed);

#endif // DMRICM_GDRCYLINDERSFIXEDRADII_H