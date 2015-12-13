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
MOT_FLOAT_TYPE cmGDRCylinders(const MOT_FLOAT_TYPE4 g,
                           const MOT_FLOAT_TYPE G,
                           const MOT_FLOAT_TYPE Delta,
                           const MOT_FLOAT_TYPE delta,
                           const MOT_FLOAT_TYPE d,
                           const MOT_FLOAT_TYPE theta,
                           const MOT_FLOAT_TYPE phi,
                           const MOT_FLOAT_TYPE gamma_k,
                           const MOT_FLOAT_TYPE gamma_beta,
                           const MOT_FLOAT_TYPE gamma_nmr_cyl);

