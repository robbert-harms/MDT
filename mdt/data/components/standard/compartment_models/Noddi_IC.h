#ifndef DMRICM_NODDIIC_H
#define DMRICM_NODDIIC_H

/**
 * Author = Robbert Harms
 * Date = 2/26/14 
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the Noddi Intra Cellular (Stick with dispersion) model.
 * If Radius is fixed to 0 the model behaves as a stick (with dispersion), if non-fixed the model behaves as a
 * cylinder (with dispersion).
 *
 * It may seem redundant to have both G/Delta/delta and b as arguments. But that is for speed reasons. b is most
 * of the time available anyway, and G/Delta/delta is only needed if R is not fixed (still it must be provided for).
 *
 * @params g from the protocol /scheme
 * @params b from the protocol /scheme
 * @params G from the protocol / scheme
 * @params Delta big delta from the protocol / scheme
 * @params delta small delta from the protocol / scheme
 * @params d parameter
 * @params theta parameter
 * @params phi parameter
 * @params kappa parameter (concentration parameter of the Watson's distribution)
 * @params R the radius of the cylinder
 * @params CLJnpZeros: the bessel root zeros used by the model function
 * @params CLJnpZerosLength: the length of the bessel roots vector CLJnpZeros
 */
MOT_FLOAT_TYPE cmNoddi_IC(const MOT_FLOAT_TYPE4 g,
                       const MOT_FLOAT_TYPE b,
                       const MOT_FLOAT_TYPE G,
                       const MOT_FLOAT_TYPE Delta,
                       const MOT_FLOAT_TYPE delta,
                       const MOT_FLOAT_TYPE d,
                       const MOT_FLOAT_TYPE theta,
                       const MOT_FLOAT_TYPE phi,
                       const MOT_FLOAT_TYPE kappa,
                       const MOT_FLOAT_TYPE R,
                       global const MOT_FLOAT_TYPE* const CLJnpZeros,
                       const int CLJnpZerosLength);
                    
#endif // DMRICM_NODDIIC_H