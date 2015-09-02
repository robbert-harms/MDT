#ifndef DMRICM_NODDIEC_CL
#define DMRICM_NODDIEC_CL

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
                       const model_float d,
                       const model_float dperp,
                       const model_float theta,
                       const model_float phi,
                       const model_float kappa){

    model_float dw_0, dw_1;
    model_float dotted = pown(dot(g, (model_float4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0)), 2);

    if(kappa > 1e-5){
	    model_float factor = sqrt(kappa)/dawson(sqrt(kappa));
	    dw_0 = (-(d - dperp) + 2 * dperp     * kappa + (d - dperp) * factor) / (2.0 * kappa);
	    dw_1 = ( (d - dperp) + 2 * (d+dperp) * kappa - (d - dperp) * factor) / (4.0 * kappa);
    }
    else{
        model_float factor = 2 * (d - dperp) * kappa;
	    dw_0 = (fma(2, dperp, d) / 3.0) + (factor/22.5) + ((factor * kappa) / 236.0);
   	    dw_1 = (fma(2, dperp, d) / 3.0) - (factor/45.0) - ((factor * kappa) / 472.0);
    }
    return exp(-b * (((dw_0 - dw_1) * dotted) + dw_1));
}

#endif // DMRICM_NODDIEC_CL