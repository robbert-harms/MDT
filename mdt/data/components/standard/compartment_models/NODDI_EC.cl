/**
 * Author = Robbert Harms
 * Date = 2/26/14
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

 /**
 * Generate the compartment model signal for the NODDI Extra Cellular compartment
 * @params g from the protocol /scheme
 * @params b from the protocol / scheme
 * @params d parameter
 * @params theta parameter
 * @params phi parameter
 * @params dperp parameter (hindered diffusivity outside the cylinders in perpendicular directions)
 * @params kappa parameter (concentration parameter of the Watson's distribution)
 */
double cmNODDI_EC(const mot_float_type4 g,
                  const mot_float_type b,
                  const mot_float_type d,
                  const mot_float_type dperp,
                  const mot_float_type theta,
                  const mot_float_type phi,
                  const mot_float_type kappa){

    const mot_float_type kappa_scaled = kappa * 10;
    mot_float_type tmp;
    mot_float_type dw_0, dw_1;

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

    return exp(-b * fma((dw_0 - dw_1),
                        pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                  sin(phi) * sin(theta), cos(theta), 0)), 2),
                        dw_1));
}
