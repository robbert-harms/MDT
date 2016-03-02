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
mot_float_type cmGDRCylindersFixedRadii(const mot_float_type4 g,
                                     const mot_float_type G,
                                     const mot_float_type Delta,
                                     const mot_float_type delta,
                                     const mot_float_type d,
                                     const mot_float_type theta,
                                     const mot_float_type phi,
                                     global const mot_float_type* const gamma_cyl_radii,
                                     global const mot_float_type* const gamma_cyl_weights,
                                     const int nmr_gamma_cyl_fixed){

    mot_float_type signal = 0;
    for(int i = 0; i < nmr_gamma_cyl_fixed; i++){
        signal += gamma_cyl_weights[i] *
                    cmCylinderGPD(g, G, Delta, delta, d, theta, phi, gamma_cyl_radii[i]);
    }
    return signal;
}

