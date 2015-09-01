#ifndef DMRICM_NODDIEC_H
#define DMRICM_NODDIEC_H

/**
 * Author = Robbert Harms
 * Date = 2/26/14 
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

 /**
 * Generate the compartment model signal for the Noddi Extra Cellular model
 * @params g from the protocol /scheme
 * @params b from the protocol / scheme
 * @params d parameter
 * @params theta parameter
 * @params phi parameter
 * @params dperp parameter (hindered diffusivity outside the cylinders in perpendicular directions)
 * @params kappa parameter (concentration parameter of the Watson's distribution)
 */
model_float cmNoddi_EC(const model_float4 g,
                  const model_float b,
                  const double d,
                  const double dperp,
                  const double theta,
                  const double phi,
                  const double kappa);

#endif // DMRICM_NODDIEC_H