#ifndef DMRICM_GDRCYLINDERSFIXEDRADII_CL
#define DMRICM_GDRCYLINDERSFIXEDRADII_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmGDRCylindersFixedRadii(const model_float4 g,
                                     const model_float G,
                                     const model_float Delta,
                                     const model_float delta,
                                     const model_float d,
                                     const model_float theta,
                                     const model_float phi,
                                     global const model_float* const gamma_cyl_radii,
                                     global const model_float* const gamma_cyl_weights,
                                     const int nmr_gamma_cyl_fixed,
                                     global const model_float* const CLJnpZeros,
                                     const int CLJnpZerosLength){

    model_float signal = 0;
    for(int i = 0; i < nmr_gamma_cyl_fixed; i++){
        signal += gamma_cyl_weights[i] *
                    cmCylinderGPD(g, G, Delta, delta, d, theta, phi, gamma_cyl_radii[i], CLJnpZeros, CLJnpZerosLength);
    }
    return signal;
}

#endif // DMRICM_GDRCYLINDERSFIXEDRADII_CL