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
                          const MOT_FLOAT_TYPE kappa){

    const MOT_FLOAT_TYPE kappa_scaled = kappa * 10;
    MOT_FLOAT_TYPE tmp;
    MOT_FLOAT_TYPE dw_0, dw_1;

    if(kappa_scaled > 1e-5){
	    tmp = sqrt(kappa_scaled)/dawson(sqrt(kappa_scaled));
	    dw_0 = ( -(d - dperp) + 2 * dperp * kappa_scaled + (d - dperp) * tmp) / (2.0 * kappa_scaled);
	    dw_1 = ( (d - dperp) + 2 * (d+dperp) * kappa_scaled - (d - dperp) * tmp) / (4.0 * kappa_scaled);
    }
    else{
        tmp = 2 * (d - dperp) * kappa_scaled;
        dw_0 = ((2 * dperp + d) / 3.0) + (tmp/22.5) + ((tmp * kappa_scaled) / 236.0);
        dw_1 = ((2 * dperp + d) / 3.0) - (tmp/45.0) - ((tmp * kappa_scaled) / 472.0);
    }

    return (MOT_FLOAT_TYPE) exp(-b * fma((dw_0 - dw_1),
                                         pown(dot(g, (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta),
                                                                sin(phi) * sin(theta), cos(theta), 0)), 2),
                                         dw_1));
}
