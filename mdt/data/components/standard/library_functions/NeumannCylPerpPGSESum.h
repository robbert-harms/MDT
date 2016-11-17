#ifndef NEUMANN_CYL_PERP_PGSE_SUM_H
#define NEUMANN_CYL_PERP_PGSE_SUM_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * This function returns the summation of the signal attenuation in perpendicular direction (LePerp)
 * for Radius R, according to the Neumann model.
 *
 * The summation is the sum over the Bessel roots up to a accuracy of 1e-8, it does not
 * calculate the complete signal for a cylinder dMRI compartment model.
 */
mot_float_type NeumannCylPerpPGSESum(const mot_float_type Delta,
                                     const mot_float_type delta,
                                     const mot_float_type d,
                                     const mot_float_type R);

#endif //NEUMANN_CYL_PERP_PGSE_SUM_H
