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
MOT_FLOAT_TYPE cmNoddi_EC(const MOT_FLOAT_TYPE4 g,
                       const MOT_FLOAT_TYPE b,
                       const MOT_FLOAT_TYPE d,
                       const MOT_FLOAT_TYPE dperp,
                       const MOT_FLOAT_TYPE theta,
                       const MOT_FLOAT_TYPE phi,
                       const MOT_FLOAT_TYPE kappa);

