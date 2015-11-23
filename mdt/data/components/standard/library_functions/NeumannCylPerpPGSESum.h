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
 * This function only returns the sum over the Bessel roots, up to a accuracy of 1e-8, it does not
 * calculate the complete signal for a cylinder model.
 */
model_float NeumannCylPerpPGSESum(const model_float Delta,
                                  const model_float delta,
                                  const model_float d,
                                  const model_float R,
                                  global const model_float* const CLJnpZeros,
                                  const int CLJnpZerosLength);

#endif //NEUMANN_CYL_PERP_PGSE_SUM_H