#ifndef DMRICM_GDRCYLINDERSFIXEDRADII_CL
#define DMRICM_GDRCYLINDERSFIXEDRADII_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmGDRCylindersFixedRadii(const double4 g,
                                const double G,
                                const double Delta,
                                const double delta,
                                const double d,
                                const double theta,
                                const double phi,
                                global const double* const gamma_cyl_radii,
                                global const double* const gamma_cyl_weights,
                                const int nmr_gamma_cyl_fixed,
                                global const double* const CLJnpZeros,
                                const int CLJnpZerosLength){

    double signal = 0;
    for(int i = 0; i < nmr_gamma_cyl_fixed; i++){
        signal += gamma_cyl_weights[i] *
                    cmCylinderGPD(g, G, Delta, delta, d, theta, phi, gamma_cyl_radii[i], CLJnpZeros, CLJnpZerosLength);
    }
    return signal;
}

#endif // DMRICM_GDRCYLINDERSFIXEDRADII_CL