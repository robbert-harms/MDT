#ifndef DMRICM_GDRCYLINDERSFIXEDRADII_CL
#define DMRICM_GDRCYLINDERSFIXEDRADII_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

MOT_FLOAT_TYPE cmGDRCylindersFixedRadii(const MOT_FLOAT_TYPE4 g,
                                     const MOT_FLOAT_TYPE G,
                                     const MOT_FLOAT_TYPE Delta,
                                     const MOT_FLOAT_TYPE delta,
                                     const MOT_FLOAT_TYPE d,
                                     const MOT_FLOAT_TYPE theta,
                                     const MOT_FLOAT_TYPE phi,
                                     global const MOT_FLOAT_TYPE* const gamma_cyl_radii,
                                     global const MOT_FLOAT_TYPE* const gamma_cyl_weights,
                                     const int nmr_gamma_cyl_fixed,
                                     global const MOT_FLOAT_TYPE* const CLJnpZeros,
                                     const int CLJnpZerosLength){

    MOT_FLOAT_TYPE signal = 0;
    for(int i = 0; i < nmr_gamma_cyl_fixed; i++){
        signal += gamma_cyl_weights[i] *
                    cmCylinderGPD(g, G, Delta, delta, d, theta, phi, gamma_cyl_radii[i], CLJnpZeros, CLJnpZerosLength);
    }
    return signal;
}

#endif // DMRICM_GDRCYLINDERSFIXEDRADII_CL