#ifndef DMRICM_CHARMEDRESTRICTED_CL
#define DMRICM_CHARMEDRESTRICTED_CL

/**
 * Author = Robbert Harms
 * Date = 2014-11-06
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmCharmedRestricted(const model_float4 g,
                           const model_float b,
                           const model_float GAMMA2_G2_delta2,
                           const model_float TE,
                           const double d,
                           const double theta,
                           const double phi){

    // gamma_cyl_radii squared
    const model_float gamma_cyl_radii_sq[] = {2.25e-12, 6.25e-12, 1.225e-11,
                                         2.025e-11, 3.025e-11, 4.224999e-11};
    const model_float gamma_cyl_weights[] = {0.0211847200855742, 0.107169623942214, 0.194400551313197,
                                        0.266676876170322, 0.214921653661151, 0.195646574827541};

    // tmp0 is also used in the loop, do not reassign it easily.
    model_float tmp0 = pown(dot(g, (model_float4)(cos(phi) * sin(theta),
                                        sin(phi) * sin(theta), cos(theta), 0.0)), 2);
    const model_float tmp1 = (-d * b * tmp0);
    const model_float tmp2 = ((-GAMMA2_G2_delta2 * (1 - tmp0)) / (d * (TE / 2.0))) * (7/96.0);
    const model_float tmp3 = (99/112.0) / (d * (TE / 2.0));

    // using tmp0 for the looping, this saves a double
    tmp0 = 0;
    // the 6 in the loop is nmr_gamma_cyl_fixed (should match the arrays above)
    for(int i = 0; i < 6; i++){
        tmp0 += gamma_cyl_weights[i]
                    * exp(tmp1 + (tmp2 * pown(gamma_cyl_radii_sq[i], 2) * (2 - (tmp3 * gamma_cyl_radii_sq[i]))));
    }
    return tmp0;
}

#endif // DMRICM_CHARMEDRESTRICTED_CL