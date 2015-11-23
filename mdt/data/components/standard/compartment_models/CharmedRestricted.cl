#ifndef DMRICM_CHARMEDRESTRICTED_CL
#define DMRICM_CHARMEDRESTRICTED_CL

/**
 * Author = Robbert Harms
 * Date = 2014-11-06
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

MOT_FLOAT_TYPE cmCharmedRestricted(const MOT_FLOAT_TYPE4 g,
                                const MOT_FLOAT_TYPE b,
                                const MOT_FLOAT_TYPE GAMMA2_G2_delta2,
                                const MOT_FLOAT_TYPE TE,
                                const MOT_FLOAT_TYPE d,
                                const MOT_FLOAT_TYPE theta,
                                const MOT_FLOAT_TYPE phi){

    // gamma_cyl_radii squared
    const MOT_FLOAT_TYPE gamma_cyl_radii_sq[] = {2.25e-12, 6.25e-12, 1.225e-11,
                                         2.025e-11, 3.025e-11, 4.224999e-11};
    const MOT_FLOAT_TYPE gamma_cyl_weights[] = {0.0211847200855742, 0.107169623942214, 0.194400551313197,
                                        0.266676876170322, 0.214921653661151, 0.195646574827541};

    // tmp0 is also used in the loop, do not reassign it easily.
    MOT_FLOAT_TYPE tmp0 = pown(dot(g, (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta),
                                        sin(phi) * sin(theta), cos(theta), 0.0)), 2);
    const MOT_FLOAT_TYPE tmp1 = (-d * b * tmp0);
    const MOT_FLOAT_TYPE tmp2 = ((-GAMMA2_G2_delta2 * (1 - tmp0)) / (d * (TE / 2.0))) * (7/96.0);
    const MOT_FLOAT_TYPE tmp3 = (99/112.0) / (d * (TE / 2.0));

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